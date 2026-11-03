# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for model_downloader.py — covers uncovered branches across all downloader classes."""

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import SecretStr

from scripts.get_huggingface_models import HfRepoAccessError
from scripts.utils.hf_common import HfErrorKind
from server.utils.core import DownloadedPacket, HttpClientError, PreDownloadPacket, SuccessDownloadPacket
from server.utils.model_downloader import (
    AdapterRegistryDownloader,
    CivitaiModelDownloader,
    HuggingFaceModelDownloader,
    HuggingFaceRepoDownloader,
    HuggingFaceRepoWithBlobsDownloader,
    ModelDownloader,
    StandardModelDownloader,
)


def make_http_error(body: str = "error") -> HttpClientError:
    return HttpClientError(message="error", status_code=500, headers=CIMultiDictProxy(CIMultiDict()), body=body)


def make_mock_config(tmp_path: Path) -> MagicMock:
    config = MagicMock()
    config.get_storage_dir.return_value = tmp_path
    config.hugging_face_token = SecretStr("hf-token")
    config.civitai_token = SecretStr("civitai-token")
    config.adapter_registry_secret = SecretStr("secret")
    return config


def make_client_session_mock(response_mock: AsyncMock) -> MagicMock:
    """Build a ClientSession mock that returns response_mock from async-with session.get(...)."""
    mock_session = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=response_mock)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_client_session = MagicMock()
    mock_client_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_client_session.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_client_session


async def _yield(*packets: Any) -> AsyncGenerator[Any]:  # pyright: ignore[reportUnusedFunction]
    for p in packets:
        yield p


def _make_hf_repo_dl(key: str = "hf-key") -> HuggingFaceRepoDownloader:
    return HuggingFaceRepoDownloader(key)


def _make_hf_model_dl() -> HuggingFaceModelDownloader:
    return HuggingFaceModelDownloader("hf-key")


def _make_hf_blobs_dl(tmp_path: Path) -> HuggingFaceRepoWithBlobsDownloader:
    return HuggingFaceRepoWithBlobsDownloader("hf-key", tmp_path / "temp")


def _make_civitai_dl() -> CivitaiModelDownloader:
    return CivitaiModelDownloader("civitai-token")


def test_create_error_msg_json_with_message_key_applies_modifier() -> None:
    dl = _make_hf_repo_dl("key")
    body = json.dumps({"message": "Invalid credentials in Authorization header"})

    result = dl.create_error_msg(body)

    assert isinstance(result, dict)
    assert "HuggingFace Token" in result["message"]


def test_create_error_msg_plain_string_applies_modifier() -> None:
    dl = _make_hf_repo_dl("key")

    result = dl.create_error_msg("Invalid credentials in Authorization header")

    assert isinstance(result, str)
    assert "HuggingFace Token" in result


def test_create_error_msg_invalid_json_falls_back_to_plain_string() -> None:
    dl = _make_hf_repo_dl("key")

    result = dl.create_error_msg("not-json")

    assert result == "not-json"


def test_create_error_msg_json_without_message_key_returns_string() -> None:
    dl = _make_hf_repo_dl("key")
    body = json.dumps({"error": "something"})

    result = dl.create_error_msg(body)

    assert isinstance(result, str)


def test_create_error_msg_restricted_model_embeds_markdown_link() -> None:
    """The WebUI renders [text](url) as a clickable link — plain text URLs are not clickable."""
    dl = _make_hf_repo_dl("key")

    result = dl.create_error_msg(
        "is restricted. You must have access to it and be authenticated to access it. Please log in.",
        model_page_url="https://huggingface.co/meta-llama/Llama-3.2-1B",
    )

    assert isinstance(result, str)
    assert "[the model page](https://huggingface.co/meta-llama/Llama-3.2-1B)" in result


def test_standard_downloader_check_url_always_true() -> None:
    dl = StandardModelDownloader()

    assert dl.check_url("https://anything.com/model.gguf") is True
    assert dl.check_url("") is True
    assert dl.check_url("relative/path") is True


@pytest.mark.asyncio
async def test_standard_downloader_download_forwards_all_packets(tmp_path: Path) -> None:
    dl = StandardModelDownloader()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    expected = [PreDownloadPacket(100), DownloadedPacket(100), SuccessDownloadPacket(model_dir)]

    async def mock_ensure(*args: Any, **kwargs: Any):
        for p in expected:
            yield p

    with patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure):
        result = [p async for p in dl.download("https://example.com/model.gguf", model_dir, temp_dir)]

    assert result == expected


def test_hf_repo_downloader_init_sets_headers_and_modifiers() -> None:
    dl = _make_hf_repo_dl()

    assert dl.headers == {"Authorization": "Bearer hf-key"}
    assert len(dl.error_msg_modifiers) == 3


@pytest.mark.parametrize(
    ("url", "expected_result"),
    [
        ("username/repo", True),
        ("https://huggingface.co/username/repo", True),
        ("https://huggingface.co/username/repo/tree/main", True),
        ("https://huggingface.co/resolve/repo", True),
        ("https://example.com/model.gguf", False),
        ("https://huggingface.co/username/repo/resolve/main/model.safetensors?download=true", False),
        ("https://huggingface.co/username/repo/resolve/main/model.gguf", False),
    ],
    ids=[
        "relative_model_id",
        "full_huggingface_url",
        "huggingface_tree_main",
        "repo_owner_named_resolve",
        "non_huggingface_url",
        "single_file_safetensors",
        "single_file_gguf",
    ],
)
def test_hf_repo_downloader_check_url(url: str, expected_result: bool) -> None:
    downloader = _make_hf_repo_dl()

    assert downloader.check_url(url) is expected_result


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_parses_file_list() -> None:
    response_data = [
        {"type": "file", "path": "model.safetensors", "size": 1000},
        {"type": "file", "path": "config.json", "size": 200},
        {"type": "directory", "path": "some-dir", "size": 0},
    ]
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_data)

    with patch("server.utils.model_downloader.ClientSession", make_client_session_mock(mock_response)):
        filenames, size = await HuggingFaceRepoDownloader.get_filenames("user/repo")

    assert filenames == ["model.safetensors", "config.json"]
    assert size == 1200


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_raises_http_client_error_on_non_200_status() -> None:
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.text = AsyncMock(return_value="Not Found")
    mock_response.headers = CIMultiDictProxy(CIMultiDict())

    with (
        patch("server.utils.model_downloader.ClientSession", make_client_session_mock(mock_response)),
        pytest.raises(HttpClientError) as exc_info,
    ):
        await HuggingFaceRepoDownloader.get_filenames("user/repo")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_network_error_propagates() -> None:
    mock_client_session = MagicMock()
    mock_client_session.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("network error"))
    mock_client_session.return_value.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("server.utils.model_downloader.ClientSession", mock_client_session),
        pytest.raises(RuntimeError, match="network error"),
    ):
        await HuggingFaceRepoDownloader.get_filenames("user/repo")


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_invalid_size_raises() -> None:
    response_data = [{"type": "file", "path": "model.bin", "size": "not-a-number"}]
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=response_data)

    with (
        patch("server.utils.model_downloader.ClientSession", make_client_session_mock(mock_response)),
        pytest.raises(ValueError, match="not-a-number"),
    ):
        await HuggingFaceRepoDownloader.get_filenames("user/repo")


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_uses_revision() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[{"type": "file", "path": "model.safetensors", "size": 1000}])
    client_session_mock = make_client_session_mock(mock_response)

    with patch("server.utils.model_downloader.ClientSession", client_session_mock):
        await HuggingFaceRepoDownloader.get_filenames("user/repo", revision="v2")

    session_mock = client_session_mock.return_value.__aenter__.return_value
    requested_url = session_mock.get.call_args.args[0]
    assert requested_url == "https://huggingface.co/api/models/user/repo/tree/v2"


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_sends_given_headers() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[{"type": "file", "path": "model.safetensors", "size": 1000}])
    client_session_mock = make_client_session_mock(mock_response)

    with patch("server.utils.model_downloader.ClientSession", client_session_mock):
        await HuggingFaceRepoDownloader.get_filenames("user/repo", headers={"Authorization": "Bearer x"})

    session_mock = client_session_mock.return_value.__aenter__.return_value
    assert session_mock.get.call_args.kwargs["headers"] == {"Authorization": "Bearer x"}


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_sends_no_headers_by_default() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[{"type": "file", "path": "model.safetensors", "size": 1000}])
    client_session_mock = make_client_session_mock(mock_response)

    with patch("server.utils.model_downloader.ClientSession", client_session_mock):
        await HuggingFaceRepoDownloader.get_filenames("user/repo")

    session_mock = client_session_mock.return_value.__aenter__.return_value
    assert session_mock.get.call_args.kwargs["headers"] == {}


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_uses_revision_in_resolve_url(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    model_dir.mkdir()

    captured_urls: list[str] = []

    async def mock_ensure(url: str, *args: Any, **kwargs: Any):
        captured_urls.append(url)
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(return_value=(["model.safetensors"], 1000))),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        packets = [p async for p in dl.download("user/repo", model_dir, temp_dir, revision="v2")]

    assert captured_urls == ["https://huggingface.co/user/repo/resolve/v2/model.safetensors"]
    assert isinstance(packets[-1], SuccessDownloadPacket)


@pytest.mark.asyncio
async def test_hf_repo_downloader_get_filenames_url_encodes_revision() -> None:
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[{"type": "file", "path": "model.safetensors", "size": 1000}])
    client_session_mock = make_client_session_mock(mock_response)

    with patch("server.utils.model_downloader.ClientSession", client_session_mock):
        await HuggingFaceRepoDownloader.get_filenames("user/repo", revision="refs/pr/1#notes")

    session_mock = client_session_mock.return_value.__aenter__.return_value
    requested_url = session_mock.get.call_args.args[0]
    assert requested_url == "https://huggingface.co/api/models/user/repo/tree/refs%2Fpr%2F1%23notes"


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_url_encodes_revision_in_resolve_url(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    model_dir.mkdir()

    captured_urls: list[str] = []

    async def mock_ensure(url: str, *args: Any, **kwargs: Any):
        captured_urls.append(url)
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(return_value=(["model.safetensors"], 1000))),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        packets = [p async for p in dl.download("user/repo", model_dir, temp_dir, revision="refs/pr/1#notes")]

    assert captured_urls == ["https://huggingface.co/user/repo/resolve/refs%2Fpr%2F1%23notes/model.safetensors"]
    assert isinstance(packets[-1], SuccessDownloadPacket)


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_extracts_model_id_from_tree_url(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    model_dir.mkdir()

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(return_value=(["model.safetensors"], 1000))),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        packets = [p async for p in dl.download("https://huggingface.co/user/repo/tree/main", model_dir, temp_dir)]

    assert isinstance(packets[-1], SuccessDownloadPacket)
    assert packets[-1].local_path == model_dir


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_yields_pre_download_packet(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    model_dir.mkdir()

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(return_value=(["model.safetensors"], 1000))),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        packets = [p async for p in dl.download("user/repo", model_dir, temp_dir)]

    assert isinstance(packets[0], PreDownloadPacket)
    assert packets[0].file_bytes_size == 1000


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_raises_http_exception_on_client_error(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    model_dir.mkdir()

    async def failing_ensure(*args: Any, **kwargs: Any):
        raise make_http_error("Invalid credentials in Authorization header")
        yield

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(return_value=(["model.gguf"], 100))),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=failing_ensure),
        pytest.raises(HTTPException) as exc_info,
    ):
        async for _ in dl.download("https://huggingface.co/user/repo", model_dir, temp_dir):
            pass

    assert exc_info.value.status_code == 500


def make_listing_error(status_code: int, body: str, error_code: str | None = None) -> HttpClientError:
    headers = CIMultiDict({"X-Error-Code": error_code}) if error_code else CIMultiDict()
    return HttpClientError(message="error", status_code=status_code, headers=CIMultiDictProxy(headers), body=body)


async def _download_with_failing_listing(dl: HuggingFaceRepoDownloader, tmp_path: Path, error: HttpClientError) -> HTTPException:
    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(side_effect=error)),
        pytest.raises(HTTPException) as exc_info,
    ):
        async for _ in dl.download("user/repo", tmp_path / "models", tmp_path / "temp"):
            pass
    return exc_info.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token", "error", "expected_fragment"),
    [
        pytest.param(
            "hf-key", make_listing_error(401, "Invalid username or password."), "token is invalid or has no access", id="401-with-token"
        ),
        pytest.param("", make_listing_error(401, "Invalid username or password."), "DF_HUGGING_FACE_TOKEN", id="401-without-token"),
        pytest.param(
            "hf-key",
            make_listing_error(404, '{"error":"Repository not found"}', "RepoNotFound"),
            "token is invalid or has no access",
            id="404-repo-not-found-with-token",
        ),
    ],
)
async def test_hf_repo_downloader_download_reports_rejected_listing_as_access_error(
    token: str, error: HttpClientError, expected_fragment: str, tmp_path: Path
) -> None:
    exc = await _download_with_failing_listing(_make_hf_repo_dl(token), tmp_path, error)

    assert exc.status_code == 404
    assert exc.detail == str(HfRepoAccessError("user/repo", None, HfErrorKind.NO_ACCESS, authenticated=bool(token)))
    assert expected_fragment in str(exc.detail)
    assert '{"error"' not in str(exc.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_code", "body", "expected_status"),
    [
        pytest.param(404, "RevisionNotFound", "Revision Not Found", 404, id="404-revision-not-found"),
        pytest.param(404, None, "Not Found", 404, id="404-without-error-code"),
        pytest.param(403, None, "Forbidden", 500, id="403"),
        pytest.param(503, None, "Service Unavailable", 500, id="5xx"),
    ],
)
async def test_hf_repo_downloader_download_other_listing_failures_are_not_reported_as_access_errors(
    status_code: int, error_code: str | None, body: str, expected_status: int, tmp_path: Path
) -> None:
    exc = await _download_with_failing_listing(_make_hf_repo_dl(), tmp_path, make_listing_error(status_code, body, error_code))

    assert exc.status_code == expected_status
    assert exc.detail == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token", "expected_headers"),
    [
        pytest.param("hf-key", {"Authorization": "Bearer hf-key"}, id="with-token"),
        pytest.param("", {}, id="without-token"),
    ],
)
async def test_hf_repo_downloader_download_passes_token_headers_to_get_filenames(
    token: str, expected_headers: dict[str, str], tmp_path: Path
) -> None:
    dl = _make_hf_repo_dl(token)
    get_filenames_mock = AsyncMock(return_value=([], 0))

    with patch.object(HuggingFaceRepoDownloader, "get_filenames", new=get_filenames_mock):
        packets = [p async for p in dl.download("user/repo", tmp_path / "models", tmp_path / "temp")]

    assert get_filenames_mock.call_args.kwargs["headers"] == expected_headers
    assert isinstance(packets[-1], SuccessDownloadPacket)


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_other_status_from_get_filenames_goes_through_raise_http_error(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"

    async def failing_get_filenames(
        model_id: str, revision: str | None = None, headers: dict[str, str] | None = None
    ) -> tuple[list[str], int]:
        raise make_http_error("Invalid credentials in Authorization header")

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(side_effect=failing_get_filenames)),
        pytest.raises(HTTPException) as exc_info,
    ):
        async for _ in dl.download("user/repo", model_dir, temp_dir):
            pass

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_http_url_not_matching_hf_pattern_uses_url_as_model_id(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    model_dir.mkdir()
    captured_model_id: list[str] = []

    async def capture(model_id: str, revision: str | None = None, headers: dict[str, str] | None = None) -> tuple[list[str], int]:
        captured_model_id.append(model_id)
        return ([], 0)

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(side_effect=capture)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        packets = [p async for p in dl.download("http://not-huggingface.com/model", model_dir, temp_dir)]

    assert captured_model_id == ["http://not-huggingface.com/model"]
    assert isinstance(packets[-1], SuccessDownloadPacket)


@pytest.mark.asyncio
async def test_hf_repo_downloader_download_filters_pre_download_packets_from_ensure(tmp_path: Path) -> None:
    dl = _make_hf_repo_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    model_dir.mkdir()

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield PreDownloadPacket(9999)
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoDownloader, "get_filenames", new=AsyncMock(return_value=(["model.safetensors"], 500))),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        packets = [p async for p in dl.download("user/repo", model_dir, temp_dir)]

    assert not any(isinstance(p, PreDownloadPacket) and p.file_bytes_size == 9999 for p in packets)
    assert any(isinstance(p, DownloadedPacket) for p in packets)


def test_hf_model_downloader_init_sets_headers_and_modifiers() -> None:
    dl = _make_hf_model_dl()

    assert dl.headers == {"Authorization": "Bearer hf-key"}
    assert len(dl.error_msg_modifiers) == 3


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://huggingface.co/user/repo/resolve/main/model.gguf", True),
        ("https://huggingface.co/user/repo/resolve/main/model.safetensors", True),
        ("https://huggingface.co/user/repo/resolve/main/model.safetensors?download=true", True),
        ("https://huggingface.co/user/repo/resolve/v1.0/sub/dir/model.bin", True),
        ("https://huggingface.co/user/repo", False),
        ("https://huggingface.co/user/repo/tree/main", False),
        ("https://example.com/model.gguf", False),
        ("https://example.com/user/repo/resolve/main/model.safetensors", False),
    ],
)
def test_hf_model_downloader_check_url(url: str, expected: bool) -> None:
    assert _make_hf_model_dl().check_url(url) is expected


@pytest.mark.asyncio
async def test_hf_model_downloader_download_strips_query_string(tmp_path: Path) -> None:
    dl = _make_hf_model_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    captured_url = None

    async def capture(url: str, *args: Any, **kwargs: Any):
        nonlocal captured_url
        captured_url = url
        yield SuccessDownloadPacket(model_dir)

    with patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=capture):
        async for _ in dl.download(
            "https://huggingface.co/user/repo/resolve/main/model.gguf?token=abc",
            model_dir,
            temp_dir,
        ):
            pass

    assert captured_url == "https://huggingface.co/user/repo/resolve/main/model.gguf"


@pytest.mark.asyncio
async def test_hf_model_downloader_download_single_file_without_query_and_with_token(tmp_path: Path) -> None:
    dl = _make_hf_model_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    calls: list[tuple[Any, ...]] = []

    async def capture(*args: Any, **kwargs: Any):
        calls.append(args)
        yield SuccessDownloadPacket(model_dir / "sdxl_lightning_8step.safetensors", "sdxl_lightning_8step.safetensors")

    with patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=capture):
        packets = [
            p
            async for p in dl.download(
                "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors?download=true",
                model_dir,
                temp_dir,
                "sdxl_lightning_8step.safetensors",
            )
        ]

    assert calls == [
        (
            "https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors",
            model_dir,
            temp_dir,
            "sdxl_lightning_8step.safetensors",
            {"Authorization": "Bearer hf-key"},
        )
    ]
    assert isinstance(packets[-1], SuccessDownloadPacket)


@pytest.mark.asyncio
async def test_hf_model_downloader_download_raises_http_exception_on_client_error(tmp_path: Path) -> None:
    dl = _make_hf_model_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"

    async def failing_ensure(*args: Any, **kwargs: Any):
        raise make_http_error("Invalid credentials in Authorization header")
        yield

    with (
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=failing_ensure),
        pytest.raises(HTTPException),
    ):
        async for _ in dl.download(
            "https://huggingface.co/user/repo/resolve/main/model.gguf",
            model_dir,
            temp_dir,
        ):
            pass


def test_hf_blobs_downloader_init_sets_headers_temp_dir_and_modifiers(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)

    assert dl.headers == {"Authorization": "Bearer hf-key"}
    assert dl.temp_dir == tmp_path / "temp"
    assert len(dl.error_msg_modifiers) == 3


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/user/repo",
        "https://example.com",
    ],
)
def test_hf_blobs_downloader_check_url_always_returns_false(tmp_path: Path, url: str) -> None:
    dl = _make_hf_blobs_dl(tmp_path)

    assert dl.check_url(url) is False


@pytest.mark.asyncio
async def test_hf_blobs_downloader_get_commit_id_parses_sha() -> None:
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value={"sha": "abc123"})

    with patch("server.utils.model_downloader.ClientSession", make_client_session_mock(mock_response)):
        result = await HuggingFaceRepoWithBlobsDownloader._get_commit_id("user/repo")  # pyright: ignore[reportPrivateUsage]

    assert result == "abc123"


@pytest.mark.asyncio
async def test_hf_blobs_downloader_get_files_handles_lfs_and_regular_files() -> None:
    response_data = [
        {
            "type": "file",
            "path": "model.gguf",
            "oid": "plain-oid",
            "size": 500,
            "lfs": {"oid": "lfs-oid", "size": 1000, "pointerSize": 10},
        },
        {"type": "file", "path": "config.json", "oid": "oid2", "size": 200},
        {"type": "directory", "path": "some-dir", "oid": "", "size": 0},
    ]
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value=response_data)

    with patch("server.utils.model_downloader.ClientSession", make_client_session_mock(mock_response)):
        files = await HuggingFaceRepoWithBlobsDownloader._get_files("user/repo")  # pyright: ignore[reportPrivateUsage]

    assert len(files) == 2
    assert files[0]["oid"] == "lfs-oid"
    assert files[0]["size"] == 1000
    assert files[1]["oid"] == "oid2"
    assert files[1]["size"] == 200


@pytest.mark.asyncio
async def test_hf_blobs_downloader_download_creates_blobs_refs_snapshots_dirs_and_ref_file(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)
    model_dir = tmp_path / "model"
    files = [{"oid": "abc123", "size": 100, "path": "model.gguf"}]

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_commit_id", new=AsyncMock(return_value="commit1")),
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_files", new=AsyncMock(return_value=files)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        [p async for p in dl.download("https://huggingface.co/user/repo", model_dir)]

    assert (model_dir / "blobs").is_dir()
    assert (model_dir / "refs").is_dir()
    assert (model_dir / "snapshots" / "commit1").is_dir()
    assert (model_dir / "refs" / "main").read_text() == "commit1"


@pytest.mark.asyncio
async def test_hf_blobs_downloader_download_creates_symlink_in_snapshot_dir(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)
    model_dir = tmp_path / "model"
    files = [{"oid": "abc123", "size": 100, "path": "model.gguf"}]

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_commit_id", new=AsyncMock(return_value="commit1")),
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_files", new=AsyncMock(return_value=files)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        [p async for p in dl.download("https://huggingface.co/user/repo", model_dir)]

    symlink = model_dir / "snapshots" / "commit1" / "model.gguf"
    assert symlink.is_symlink()


@pytest.mark.asyncio
async def test_hf_blobs_downloader_download_skips_symlink_creation_when_symlink_exists(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)
    model_dir = tmp_path / "model"
    files = [{"oid": "abc123", "size": 100, "path": "model.gguf"}]

    symlink_dir = model_dir / "snapshots" / "commit1"
    blobs_dir = model_dir / "blobs"
    symlink_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)
    blob_file = blobs_dir / "abc123"
    blob_file.write_bytes(b"data")
    (symlink_dir / "model.gguf").symlink_to(blob_file)

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_commit_id", new=AsyncMock(return_value="commit1")),
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_files", new=AsyncMock(return_value=files)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        [p async for p in dl.download("https://huggingface.co/user/repo", model_dir)]

    symlink = model_dir / "snapshots" / "commit1" / "model.gguf"
    assert symlink.is_symlink()


@pytest.mark.asyncio
async def test_hf_blobs_downloader_download_yields_success_packet_last(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)
    model_dir = tmp_path / "model"
    files = [{"oid": "abc123", "size": 100, "path": "model.gguf"}]

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_commit_id", new=AsyncMock(return_value="commit1")),
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_files", new=AsyncMock(return_value=files)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        packets = [p async for p in dl.download("https://huggingface.co/user/repo", model_dir)]

    assert isinstance(packets[-1], SuccessDownloadPacket)
    assert packets[-1].local_path == model_dir


@pytest.mark.asyncio
async def test_hf_blobs_downloader_download_filter_out_other_modelfiles_removes_pytorch_and_flax(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)
    model_dir = tmp_path / "model"
    files = [
        {"oid": "oid1", "size": 100, "path": "model.safetensors"},
        {"oid": "oid2", "size": 100, "path": "pytorch_model.bin"},
        {"oid": "oid3", "size": 100, "path": "flax_model.msgpack"},
        {"oid": "oid4", "size": 100, "path": "config.json"},
    ]
    downloaded_paths: list[str] = []

    async def capture(url: str, *args: Any, **kwargs: Any):
        downloaded_paths.append(url)
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_commit_id", new=AsyncMock(return_value="commit1")),
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_files", new=AsyncMock(return_value=files)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=capture),
    ):
        [p async for p in dl.download("https://huggingface.co/user/repo", model_dir, filter_out_other_modelfiles=True)]

    assert all("pytorch_model.bin" not in url for url in downloaded_paths)
    assert all("flax_model.msgpack" not in url for url in downloaded_paths)
    assert any("model.safetensors" in url for url in downloaded_paths)
    assert any("config.json" in url for url in downloaded_paths)


@pytest.mark.asyncio
async def test_hf_blobs_downloader_download_raises_http_exception_on_client_error(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)
    model_dir = tmp_path / "model"
    files = [{"oid": "abc123", "size": 100, "path": "model.gguf"}]

    async def failing_ensure(*args: Any, **kwargs: Any):
        raise make_http_error("Invalid credentials in Authorization header")
        yield

    with (
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_commit_id", new=AsyncMock(return_value="commit1")),
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_files", new=AsyncMock(return_value=files)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=failing_ensure),
        pytest.raises(HTTPException),
    ):
        async for _ in dl.download("https://huggingface.co/user/repo", model_dir):
            pass


@pytest.mark.asyncio
async def test_hf_blobs_downloader_download_does_not_overwrite_existing_ref_file(tmp_path: Path) -> None:
    dl = _make_hf_blobs_dl(tmp_path)
    model_dir = tmp_path / "model"
    files = [{"oid": "abc123", "size": 100, "path": "model.gguf"}]
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_file = refs_dir / "main"
    ref_file.write_text("old-commit")

    async def mock_ensure(*args: Any, **kwargs: Any):
        yield DownloadedPacket(100)

    with (
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_commit_id", new=AsyncMock(return_value="new-commit")),
        patch.object(HuggingFaceRepoWithBlobsDownloader, "_get_files", new=AsyncMock(return_value=files)),
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=mock_ensure),
    ):
        [p async for p in dl.download("https://huggingface.co/user/repo", model_dir)]

    assert ref_file.read_text() == "old-commit"


def test_civitai_downloader_init_sets_token_and_modifiers() -> None:
    dl = _make_civitai_dl()

    assert dl.token == "civitai-token"
    assert len(dl.error_msg_modifiers) == 1


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://civitai.com/models/123", True),
        ("https://huggingface.co/user/repo", False),
        ("https://example.com/model.gguf", False),
    ],
)
def test_civitai_downloader_check_url(url: str, expected: bool) -> None:
    assert _make_civitai_dl().check_url(url) is expected


def test_civitai_downloader_add_token_to_url_appends_token() -> None:
    dl = _make_civitai_dl()

    result = dl.add_token_to_url("https://civitai.com/api/download/models/123")

    assert "token=civitai-token" in result


@pytest.mark.asyncio
async def test_civitai_downloader_download_passes_token_appended_url(tmp_path: Path) -> None:
    dl = _make_civitai_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"
    captured_url = None

    async def capture(url: str, *args: Any, **kwargs: Any):
        nonlocal captured_url
        captured_url = url
        yield SuccessDownloadPacket(model_dir)

    with patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=capture):
        async for _ in dl.download("https://civitai.com/api/download/models/123", model_dir, temp_dir):
            pass

    assert captured_url is not None
    assert "token=civitai-token" in captured_url


@pytest.mark.asyncio
async def test_civitai_downloader_download_raises_http_exception_on_client_error(tmp_path: Path) -> None:
    dl = _make_civitai_dl()
    model_dir = tmp_path / "models"
    temp_dir = tmp_path / "temp"

    async def failing_ensure(*args: Any, **kwargs: Any):
        raise make_http_error("The creator of this asset requires you to be logged in to download it")
        yield

    with (
        patch("server.utils.model_downloader.Utils.ensure_model_downloaded", side_effect=failing_ensure),
        pytest.raises(HTTPException) as exc_info,
    ):
        async for _ in dl.download("https://civitai.com/api/download/models/123", model_dir, temp_dir):
            pass

    assert exc_info.value.status_code == 500
    assert "DF_CIVITAI_TOKEN" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_adapter_registry_downloader_download_generic_exception_cleans_temp_and_reraises(tmp_path: Path) -> None:
    dl = AdapterRegistryDownloader(secret="test-secret")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    async def fail_with_runtime(url: str, file_path: Path, headers: dict[str, str]):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"partial")
        raise RuntimeError("unexpected network failure")
        yield

    with (
        patch("server.utils.model_downloader.download_file", side_effect=fail_with_runtime),
        pytest.raises(RuntimeError, match="unexpected network failure"),
    ):
        async for _ in dl.download("http://registry.local:5000/adapter.gguf", model_dir, temp_dir):
            pass

    if temp_dir.exists():
        assert list(temp_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_adapter_registry_downloader_download_empty_file_raises_runtime_error(tmp_path: Path) -> None:
    dl = AdapterRegistryDownloader(secret="test-secret")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    async def write_empty(url: str, file_path: Path, headers: dict[str, str]):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"")
        yield PreDownloadPacket(0)

    with (
        patch("server.utils.model_downloader.download_file", side_effect=write_empty),
        pytest.raises(RuntimeError, match="missing or empty"),
    ):
        async for _ in dl.download("http://registry.local:5000/adapter.gguf", model_dir, temp_dir):
            pass


@pytest.mark.asyncio
async def test_adapter_registry_downloader_download_generic_exception_without_temp_file_still_reraises(tmp_path: Path) -> None:
    dl = AdapterRegistryDownloader(secret="test-secret")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    async def raise_immediately(url: str, file_path: Path, headers: dict[str, str]):
        raise RuntimeError("instant failure, no file created")
        yield

    with (
        patch("server.utils.model_downloader.download_file", side_effect=raise_immediately),
        pytest.raises(RuntimeError, match="instant failure"),
    ):
        async for _ in dl.download("http://registry.local:5000/adapter.gguf", model_dir, temp_dir):
            pass


@pytest.mark.asyncio
async def test_adapter_registry_downloader_download_missing_temp_file_raises_runtime_without_unlink(tmp_path: Path) -> None:
    dl = AdapterRegistryDownloader(secret="test-secret")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    async def yield_without_creating_file(url: str, file_path: Path, headers: dict[str, str]):
        yield PreDownloadPacket(0)

    with (
        patch("server.utils.model_downloader.download_file", side_effect=yield_without_creating_file),
        pytest.raises(RuntimeError, match="missing or empty"),
    ):
        async for _ in dl.download("http://registry.local:5000/adapter.gguf", model_dir, temp_dir):
            pass


def test_model_downloader_create_downloaders_sets_all_fields(tmp_path: Path) -> None:
    config = make_mock_config(tmp_path)

    md = ModelDownloader(config)

    assert md.temp_dir == tmp_path / "temp"
    assert isinstance(md.standard_downloader, StandardModelDownloader)
    assert len(md.custom_downloaders) == 3
    assert isinstance(md.custom_downloaders[0], HuggingFaceRepoDownloader)
    assert isinstance(md.custom_downloaders[1], HuggingFaceModelDownloader)
    assert isinstance(md.custom_downloaders[2], CivitaiModelDownloader)
    assert isinstance(md.hugging_face_repo_with_blobs_downloader, HuggingFaceRepoWithBlobsDownloader)


def test_model_downloader_get_hugging_face_token_returns_config_value(tmp_path: Path) -> None:
    config = make_mock_config(tmp_path)

    md = ModelDownloader(config)

    assert md.get_hugging_face_token(config) == "hf-token"


def test_model_downloader_get_civitai_token_returns_config_value(tmp_path: Path) -> None:
    config = make_mock_config(tmp_path)

    md = ModelDownloader(config)

    assert md.get_civitai_token(config) == "civitai-token"


@pytest.mark.asyncio
async def test_model_downloader_download_routes_civitai_url_to_civitai_downloader(tmp_path: Path) -> None:
    config = make_mock_config(tmp_path)
    md = ModelDownloader(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    routed_to: list[str] = []

    async def mock_civitai_download(url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None):
        routed_to.append("civitai")
        yield SuccessDownloadPacket(model_dir)

    md.custom_downloaders[2].download = mock_civitai_download

    async for _ in md.download("https://civitai.com/api/download/models/123", model_dir):
        pass

    assert routed_to == ["civitai"]


@pytest.mark.asyncio
async def test_model_downloader_download_falls_back_to_standard_downloader_for_unknown_url(tmp_path: Path) -> None:
    config = make_mock_config(tmp_path)
    md = ModelDownloader(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    routed_to: list[str] = []

    async def mock_standard_download(url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None):
        routed_to.append("standard")
        yield SuccessDownloadPacket(model_dir)

    md.standard_downloader.download = mock_standard_download

    async for _ in md.download("https://example.com/model.gguf", model_dir):
        pass

    assert routed_to == ["standard"]


@pytest.mark.asyncio
async def test_model_downloader_download_routes_huggingface_gguf_url_to_hf_model_downloader(tmp_path: Path) -> None:
    config = make_mock_config(tmp_path)
    md = ModelDownloader(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    routed_to: list[str] = []

    async def mock_hf_model_download(url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None):
        routed_to.append("hf_model")
        yield SuccessDownloadPacket(model_dir)

    md.custom_downloaders[1].download = mock_hf_model_download

    async for _ in md.download("https://huggingface.co/user/repo/resolve/main/model.gguf", model_dir):
        pass

    assert routed_to == ["hf_model"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step.safetensors?download=true", "hf_model"),
        ("https://huggingface.co/ByteDance/SDXL-Lightning", "hf_repo"),
        ("https://huggingface.co/ByteDance/SDXL-Lightning/tree/main", "hf_repo"),
        ("ByteDance/SDXL-Lightning", "hf_repo"),
    ],
    ids=["single_file", "repository_url", "tree_url", "bare_id"],
)
async def test_model_downloader_download_routes_huggingface_urls(tmp_path: Path, url: str, expected: str) -> None:
    config = make_mock_config(tmp_path)
    md = ModelDownloader(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    routed_to: list[str] = []

    def mock_download(name: str):
        async def download(url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None):
            routed_to.append(name)
            yield SuccessDownloadPacket(model_dir)

        return download

    md.custom_downloaders[0].download = mock_download("hf_repo")
    md.custom_downloaders[1].download = mock_download("hf_model")

    async for _ in md.download(url, model_dir):
        pass

    assert routed_to == [expected]


@pytest.mark.asyncio
async def test_model_downloader_download_logs_progress_percentage(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = make_mock_config(tmp_path)
    md = ModelDownloader(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    async def mock_standard_download(url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None):
        yield PreDownloadPacket(100)
        yield DownloadedPacket(5)  # below the 10% log step, should not log yet
        yield DownloadedPacket(45)  # crosses 50%, should log
        yield DownloadedPacket(50)  # crosses 100%, should log
        yield SuccessDownloadPacket(model_dir)

    md.standard_downloader.download = mock_standard_download

    with caplog.at_level("INFO", logger="uvicorn"):
        async for _ in md.download("https://example.com/model.gguf", model_dir):
            pass

    assert "5%" not in caplog.text
    assert "50%" in caplog.text
    assert "100%" in caplog.text
    assert "Downloaded https://example.com/model.gguf" in caplog.text


@pytest.mark.asyncio
async def test_model_downloader_download_does_not_log_progress_when_size_unknown(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = make_mock_config(tmp_path)
    md = ModelDownloader(config)
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    async def mock_standard_download(url: str, model_dir: Path, temp_dir: Path, filename: str | None = None, revision: str | None = None):
        yield PreDownloadPacket(0)
        yield DownloadedPacket(50)
        yield SuccessDownloadPacket(model_dir)

    md.standard_downloader.download = mock_standard_download

    with caplog.at_level("INFO", logger="uvicorn"):
        async for _ in md.download("https://example.com/model.gguf", model_dir):
            pass

    assert "%" not in caplog.text
