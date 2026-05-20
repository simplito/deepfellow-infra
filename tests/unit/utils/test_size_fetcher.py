# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.utils.size_fetcher import (
    _fetch_file_size_bytes,  # pyright: ignore[reportPrivateUsage]
    _fetch_hf_size_bytes,  # pyright: ignore[reportPrivateUsage]
    _fetch_ollama_size_bytes,  # pyright: ignore[reportPrivateUsage]
    _hf_id_from_ref,  # pyright: ignore[reportPrivateUsage]
    _is_hf_ref,  # pyright: ignore[reportPrivateUsage]
    _parse_ollama_tag_size,  # pyright: ignore[reportPrivateUsage]
    fetch_file_size_from_url,
    fetch_huggingface_model_size,
    fetch_ollama_model_size,
    fetch_ollama_modelfile_size,
    fetch_ollama_ref_bytes,
    fmt_size,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0.0 B"),
        (512, "512.0 B"),
        (1023, "1023.0 B"),
        (1024, "1.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1024**3, "1.0 GB"),
        (1024**4, "1.0 TB"),
        (1024**5, "1.0 PB"),
    ],
)
def test_fmt_size(n: int, expected: str) -> None:
    assert fmt_size(n) == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("google/gemma", True),
        ("llama3", False),
        ("llama3:latest", False),
        ("huggingface.co/google/gemma", True),
        ("hf.co/google/gemma", True),
        ("/absolute/path", False),
    ],
)
def test_is_hf_ref(ref: str, expected: bool) -> None:
    assert _is_hf_ref(ref) == expected


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("google/gemma", "google/gemma"),
        ("https://huggingface.co/google/gemma", "google/gemma"),
        ("https://hf.co/google/gemma", "google/gemma"),
    ],
)
def test_hf_id_from_ref(ref: str, expected: str) -> None:
    assert _hf_id_from_ref(ref) == expected


def _make_ollama_html(entries: list[tuple[str, str]]) -> str:
    """Build minimal fake HTML matching the ollama.com/library/{name}/tags structure."""
    parts = []
    for href, size in entries:
        parts.append(f'group px-4 py-3 ...<a href="{href}">link</a><p class="col-span-2">{size}</p>')
    # Prefix with something so split("group px-4 py-3")[1:] works
    return "HEADER " + " ".join(parts)


@pytest.mark.parametrize(
    ("entries", "tag", "expected"),
    [
        ([("/library/llama3:7b", "4.7 GB"), ("/library/llama3:latest", "4.7 GB")], "7b", "4.7 GB"),
        ([("/library/llama3:latest", "4.7 GB")], "latest", "4.7 GB"),
        ([("/library/llama3:7b", "4.7 GB")], "13b", None),
        ([], "latest", None),
    ],
)
def test_parse_ollama_tag_size(entries: list[tuple[str, str]], tag: str, expected: str | None) -> None:
    html = _make_ollama_html(entries)

    result = _parse_ollama_tag_size(html, tag)

    assert result == expected


def test_parse_ollama_tag_size_part_without_href_is_skipped() -> None:
    # First part has no matching href → continue; second part contains the tag
    html = (
        "HEADER "
        'group px-4 py-3 <p class="col-span-2">no-href-here</p>'
        ' group px-4 py-3 <a href="/library/llama3:7b">link</a>'
        '<p class="col-span-2">4.7 GB</p>'
    )
    result = _parse_ollama_tag_size(html, "7b")
    assert result == "4.7 GB"


@pytest.mark.asyncio
@pytest.mark.parametrize("ref", ["/foo/bar", "./foo/bar"])
async def test_fetch_ollama_ref_bytes_path_returns_none(ref: str) -> None:
    result = await fetch_ollama_ref_bytes(ref)

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("patch_target", "ref", "return_value"),
    [
        ("server.utils.size_fetcher._fetch_hf_size_bytes", "google/gemma", 1000),
        ("server.utils.size_fetcher._fetch_ollama_size_bytes", "llama3", 2000),
        ("server.utils.size_fetcher._fetch_ollama_size_bytes", "llama3:latest", 3000),
    ],
)
async def test_fetch_ollama_ref_bytes_dispatches(patch_target: str, ref: str, return_value: int) -> None:
    with patch(patch_target, new=AsyncMock(return_value=return_value)) as mock:
        result = await fetch_ollama_ref_bytes(ref)

    assert result == return_value
    assert mock.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mock_kwargs", "expected"),
    [
        ({"return_value": 1000}, fmt_size(1000)),
        ({"return_value": None}, None),
        ({"side_effect": Exception("network error")}, None),
    ],
)
async def test_fetch_file_size_from_url(mock_kwargs: dict[str, Any], expected: str | None) -> None:
    with patch("server.utils.size_fetcher._fetch_file_size_bytes", new=AsyncMock(**mock_kwargs)):
        result = await fetch_file_size_from_url("https://example.com/file")

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mock_kwargs", "expected"),
    [
        ({"return_value": 1024**3}, "1.0 GB"),
        ({"return_value": None}, None),
        ({"side_effect": Exception("fail")}, None),
    ],
)
async def test_fetch_huggingface_model_size(mock_kwargs: dict[str, Any], expected: str | None) -> None:
    with patch("server.utils.size_fetcher._fetch_hf_size_bytes", new=AsyncMock(**mock_kwargs)):
        result = await fetch_huggingface_model_size("google/gemma")

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mock_kwargs", "expected"),
    [
        ({"return_value": 1024}, "1.0 KB"),
        ({"return_value": None}, None),
        ({"side_effect": Exception("fail")}, None),
    ],
)
async def test_fetch_ollama_model_size(mock_kwargs: dict[str, Any], expected: str | None) -> None:
    with patch("server.utils.size_fetcher._fetch_ollama_size_bytes", new=AsyncMock(**mock_kwargs)):
        result = await fetch_ollama_model_size("llama3")

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "modelfile",
    [
        "",
        "PARAMETER num_ctx 4096\nSYSTEM You are helpful.",
    ],
)
async def test_fetch_ollama_modelfile_size_returns_none_without_refs(modelfile: str) -> None:
    result = await fetch_ollama_modelfile_size(modelfile)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mock_kwargs", "modelfile", "expected"),
    [
        ({"return_value": 1024}, "FROM llama3\n", "1.0 KB"),
        ({"side_effect": [1024, 2048]}, "FROM llama3\nADAPTER ./adapter.bin\n", fmt_size(1024 + 2048)),
        ({"side_effect": [Exception("fail"), 512]}, "FROM llama3\nADAPTER ./adapter.bin\n", fmt_size(512)),
        ({"return_value": None}, "FROM llama3\n", None),
    ],
)
async def test_fetch_ollama_modelfile_size(mock_kwargs: dict[str, Any], modelfile: str, expected: str | None) -> None:
    with patch("server.utils.size_fetcher.fetch_ollama_ref_bytes", new=AsyncMock(**mock_kwargs)):
        result = await fetch_ollama_modelfile_size(modelfile)

    assert result == expected


def _make_session_mock(resp_mock: MagicMock, method: str = "get") -> MagicMock:
    """Build a mock for aiohttp.ClientSession that yields resp_mock."""
    request_cm = MagicMock()
    request_cm.__aenter__ = AsyncMock(return_value=resp_mock)
    request_cm.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    getattr(mock_session, method).return_value = request_cm
    session_cm = MagicMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=session_cm)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "content_length", "expected"),
    [
        (200, "12345", 12345),
        (200, None, None),
        (404, None, None),
    ],
)
async def test_fetch_file_size_bytes(status: int, content_length: str | None, expected: int | None) -> None:
    resp = MagicMock()
    resp.status = status
    resp.headers.get.return_value = content_length

    with patch("server.utils.size_fetcher.aiohttp.ClientSession", _make_session_mock(resp, "head")):
        result = await _fetch_file_size_bytes("https://example.com/file")

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("json_response", "expected"),
    [
        ({"siblings": [{"size": 1024}, {"size": 2048}]}, 3072),
        ({"siblings": []}, None),
    ],
)
async def test_fetch_hf_size_bytes(json_response: dict[str, Any], expected: int | None) -> None:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = AsyncMock(return_value=json_response)

    with patch("server.utils.size_fetcher.aiohttp.ClientSession", _make_session_mock(resp, "get")):
        result = await _fetch_hf_size_bytes("google/gemma")

    assert result == expected


def _make_ollama_html_with_tag(tag: str, size_str: str) -> str:
    return f'HEADER group px-4 py-3 <a href="/library/llama3:{tag}">link</a><p class="col-span-2">{size_str}</p>'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("html_tag", "ref"),
    [
        ("7b", "llama3:7b"),
        ("latest", "llama3"),
    ],
)
async def test_fetch_ollama_size_bytes_returns_positive(html_tag: str, ref: str) -> None:
    html = _make_ollama_html_with_tag(html_tag, "4.7 GB")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = AsyncMock(return_value=html)

    with patch("server.utils.size_fetcher.aiohttp.ClientSession", _make_session_mock(resp, "get")):
        result = await _fetch_ollama_size_bytes(ref)

    assert result is not None
    assert result > 0


@pytest.mark.asyncio
async def test_fetch_ollama_size_bytes_returns_none_when_tag_not_found() -> None:
    html = _make_ollama_html_with_tag("7b", "4.7 GB")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = AsyncMock(return_value=html)

    with patch("server.utils.size_fetcher.aiohttp.ClientSession", _make_session_mock(resp, "get")):
        result = await _fetch_ollama_size_bytes("llama3:13b")

    assert result is None
