# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Inference through the public API, one check per modality.

Checks look at the shape of the answer plus a cheap marker that the intended backend produced it (a file header, a
known word, a score order) - never at output quality.
"""

import base64
import io
import json
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from tests.e2e.client import E2EClient, E2EError
from tests.e2e.scenarios import BROKEN_CHAT_APIS, Modality, Scenario

FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_SAMPLE = FIXTURES / "speech-sample.wav"
PROMPT = "Say hello."
MESSAGES = [{"role": "user", "content": PROMPT}]
DOCUMENT_PDF = FIXTURES / "document-sample.pdf"
DOCUMENT_DOCX = FIXTURES / "document-sample.docx"
# Text both fixtures contain, so a conversion that returns an empty or wrong document fails.
DOCUMENT_PHRASE = "release checklist"


def _sse_data(response: httpx.Response) -> list[str]:
    events = [line.removeprefix("data:").strip() for line in response.text.splitlines() if line.startswith("data:")]
    assert events, response.text[:2000]
    return events


def _chat_completions(client: E2EClient, model: str, stream: bool) -> str:
    response = client.post_api("/v1/chat/completions", json={"model": model, "messages": MESSAGES, "max_tokens": 16, "stream": stream})
    if not stream:
        return response.json()["choices"][0]["message"]["content"]
    events = _sse_data(response)
    assert events[-1] == "[DONE]", events[-3:]
    return "".join(choice["delta"].get("content") or "" for event in events[:-1] for choice in json.loads(event)["choices"])


def _completions(client: E2EClient, model: str, stream: bool) -> str:
    response = client.post_api("/v1/completions", json={"model": model, "prompt": PROMPT, "max_tokens": 16, "stream": stream})
    if not stream:
        return response.json()["choices"][0]["text"]
    events = _sse_data(response)
    assert events[-1] == "[DONE]", events[-3:]
    return "".join(choice.get("text") or "" for event in events[:-1] for choice in json.loads(event)["choices"])


def _responses(client: E2EClient, model: str, stream: bool) -> str:
    response = client.post_api("/v1/responses", json={"model": model, "input": PROMPT, "max_output_tokens": 16, "stream": stream})
    if not stream:
        output = response.json()["output"]
        return "".join(part.get("text") or "" for item in output if item["type"] == "message" for part in item["content"])
    events = [json.loads(event) for event in _sse_data(response)]
    assert events[-1]["type"] == "response.completed", events[-3:]
    return "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")


def _messages(client: E2EClient, model: str, stream: bool) -> str:
    response = client.post_api("/v1/messages", json={"model": model, "messages": MESSAGES, "max_tokens": 16, "stream": stream})
    if not stream:
        return "".join(block.get("text") or "" for block in response.json()["content"] if block["type"] == "text")
    events = [json.loads(event) for event in _sse_data(response)]
    assert events[-1]["type"] == "message_stop", events[-3:]
    return "".join(event["delta"].get("text") or "" for event in events if event["type"] == "content_block_delta")


def _ollama_chat(client: E2EClient, model: str, stream: bool) -> str:
    body = {"model": model, "messages": MESSAGES, "stream": stream, "options": {"num_predict": 16}}
    response = client.post_api("/api/chat", json=body)
    if not stream:
        return response.json()["message"]["content"]
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert lines, response.text[:2000]
    assert lines[-1].get("done") is True, lines[-3:]
    return "".join(line["message"]["content"] for line in lines if "message" in line)


# The chat-like APIs Infra proxies, each in its own wire format. A model is checked on every one it advertises.
CHAT_APIS: dict[str, Callable[[E2EClient, str, bool], str]] = {
    "/v1/chat/completions": _chat_completions,
    "/v1/completions": _completions,
    "/v1/responses": _responses,
    "/v1/messages": _messages,
    "/api/chat": _ollama_chat,
}


def check_chat(client: E2EClient, scenario: Scenario) -> None:
    """Every chat API the model advertises, plain and streamed: each is its own proxy path in Infra."""
    served = client.served_model(scenario.served_model)
    advertised = served["props"]["endpoints"] if served and served.get("props") else ["/v1/chat/completions"]
    paths = [path for path in CHAT_APIS if path in advertised]
    assert paths, f"{scenario.served_model} advertises no chat API: {advertised}"
    for path in paths:
        if reason := BROKEN_CHAT_APIS.get((scenario.service_id, path)):
            _expect_broken(client, scenario, path, reason)
            continue
        for stream in (False, True):
            try:
                text = CHAT_APIS[path](client, scenario.served_model, stream)
            except Exception as e:
                e.add_note(f"{path} stream={stream}")
                raise
            assert text.strip(), f"{path} stream={stream} returned no text"


def _expect_broken(client: E2EClient, scenario: Scenario, path: str, reason: str) -> None:
    try:
        CHAT_APIS[path](client, scenario.served_model, False)
    except E2EError:
        return
    msg = f"{path} on {scenario.service_id} works now - remove it from BROKEN_CHAT_APIS (was: {reason})"
    raise AssertionError(msg)


def check_embeddings(client: E2EClient, scenario: Scenario) -> None:
    # A list input, as Server sends document chunks in batches: one vector per text, in order.
    texts = ["Hello, world.", "Paris is the capital of France."]
    data = client.post_api("/v1/embeddings", json={"model": scenario.served_model, "input": texts}).json()
    assert [item["index"] for item in data["data"]] == list(range(len(texts))), data
    for item in data["data"]:
        vector: list[object] = item["embedding"]
        assert vector, data
        assert all(isinstance(x, int | float) for x in vector), data
        if scenario.dimensions:
            assert len(vector) == scenario.dimensions, f"{len(vector)} dimensions, expected {scenario.dimensions}"
    assert data["data"][0]["embedding"] != data["data"][1]["embedding"], "both texts got the same vector"


def check_rerank(client: E2EClient, scenario: Scenario) -> None:
    documents = ["Paris is the capital of France.", "Bananas are yellow.", "The Eiffel Tower is in Paris."]
    body = {"model": scenario.served_model, "query": "What is the capital of France?", "documents": documents}
    data = client.post_api("/v1/rerank", json=body).json()
    results = data["results"]
    assert len(results) == len(documents), data
    assert sorted(r["index"] for r in results) == list(range(len(documents))), data
    assert all(isinstance(r["relevance_score"], int | float) for r in results), data
    scores = {r["index"]: r["relevance_score"] for r in results}
    assert scores[0] > scores[1], f"the Paris answer should outscore the banana one: {scores}"


def check_image(client: E2EClient, scenario: Scenario) -> None:
    body = {"model": scenario.served_model, "prompt": "a red circle", "n": 1, "size": "256x256", "response_format": "b64_json"}
    data = client.post_api("/v1/images/generations", json=body).json()
    image = base64.b64decode(data["data"][0]["b64_json"], validate=True)
    assert image.startswith((b"\x89PNG", b"\xff\xd8\xff")) or image[8:12] == b"WEBP", f"not an image: {image[:16]!r}"


def check_speech(client: E2EClient, scenario: Scenario) -> None:
    response = client.post_api("/v1/audio/speech", json={"model": scenario.served_model, "input": "Hello, world."})
    audio = response.content
    # WAV, MP3 (with or without an ID3 tag), Ogg/Opus, FLAC.
    is_audio = audio.startswith((b"RIFF", b"ID3", b"OggS", b"fLaC")) or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    assert is_audio, f"not audio: {audio[:16]!r}, content-type {response.headers.get('content-type')}"


def check_transcription(client: E2EClient, scenario: Scenario) -> None:
    with SPEECH_SAMPLE.open("rb") as f:
        response = client.post_api(
            "/v1/audio/transcriptions", data={"model": scenario.served_model}, files={"file": (SPEECH_SAMPLE.name, f, "audio/wav")}
        )
    data = response.json()
    # The sample says "Hello, this is a test of the transcription service."; a tiny model may garble some of it.
    words = set(data["text"].lower().replace(",", " ").replace(".", " ").split())
    assert words & {"hello", "test", "transcription", "service"}, data


def check_decision(client: E2EClient, scenario: Scenario) -> None:
    # One question of each kind; Ollama rejects questions without instructions.
    questions = {
        "label": {"type": "choice", "instructions": "Which team handles this?", "criteria": {"payments": None, "shipping": None}},
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "severity": {"type": "score", "instructions": "How severe is this?", "criteria": ["low", "high"]},
    }
    body = {"model": scenario.served_model, "state": "The card payment failed twice at checkout.", "questions": questions}
    data = client.post_api("/v1/systemone", json=body).json()
    answers = data["answers"]
    assert set(answers) == set(questions), data
    assert answers["label"]["choice"] in questions["label"]["criteria"], data
    assert isinstance(answers["urgent"]["noul"], int | float), data
    assert isinstance(answers["severity"]["score"], int | float), data


def _archive_text(archive: bytes, member: str) -> str:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        return zf.read(member).decode()


def check_document(client: E2EClient, scenario: Scenario) -> None:
    """Both ways Server converts files through Infra: synchronously with chunks, and as a background job."""
    base = f"/custom/{(scenario.model_spec or {}).get('prefix', scenario.model_id)}"

    with DOCUMENT_PDF.open("rb") as f:
        archive = client.post_api(
            f"{base}/convert", params={"chunk": "true"}, files={"file": (DOCUMENT_PDF.name, f, "application/pdf")}
        ).content
    assert DOCUMENT_PHRASE in _archive_text(archive, "document.json").lower()
    chunks = json.loads(_archive_text(archive, "chunks.json"))
    assert chunks["count"] > 0, chunks
    assert DOCUMENT_PHRASE in json.dumps(chunks["chunks"]).lower(), chunks

    docx_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    job_ids = [_convert_job(client, base, DOCUMENT_DOCX, docx_type), _convert_job(client, base, DOCUMENT_PDF, "application/pdf")]
    # Two body-less 204s in a row once hung the whole proxy (DFINFRA-396); Infra must still answer afterwards.
    for job_id in job_ids:
        assert client.delete_api(f"{base}/convert/jobs/{job_id}").status_code == 204
    assert client.health(), "Infra stopped answering after deleting conversion jobs"


def _convert_job(client: E2EClient, base: str, document: Path, content_type: str) -> str:
    with document.open("rb") as f:
        job = client.post_api(f"{base}/convert/jobs", files={"file": (document.name, f, content_type)}).json()
    deadline = time.monotonic() + client.settings.inference_timeout
    while job["status"] in ("queued", "running"):
        assert time.monotonic() < deadline, f"conversion job still {job['status']} after {client.settings.inference_timeout:.0f}s"
        time.sleep(2)
        job = client.get_api(f"{base}/convert/jobs/{job['id']}").json()
    assert job["status"] == "succeeded", job
    archive = client.get_api(f"{base}/convert/jobs/{job['id']}/result").content
    assert DOCUMENT_PHRASE in _archive_text(archive, "document.json").lower()
    return str(job["id"])


CHECKS: dict[Modality, Callable[[E2EClient, Scenario], None]] = {
    "chat": check_chat,
    "embeddings": check_embeddings,
    "rerank": check_rerank,
    "image": check_image,
    "speech": check_speech,
    "transcription": check_transcription,
    "decision": check_decision,
    "document": check_document,
}


def check_inference(client: E2EClient, scenario: Scenario) -> None:
    CHECKS[scenario.modality](client, scenario)
