# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for AdapterRegistryDownloader."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from multidict import CIMultiDict, CIMultiDictProxy

from server.utils.core import DownloadedPacket, HttpClientError, PreDownloadPacket, SuccessDownloadPacket
from server.utils.model_downloader import AdapterRegistryDownloader


@pytest.fixture
def downloader() -> AdapterRegistryDownloader:
    return AdapterRegistryDownloader(url="http://registry.local:5000", secret="test-secret")


@pytest.fixture
def downloader_no_secret() -> AdapterRegistryDownloader:
    return AdapterRegistryDownloader(url="http://registry.local:5000", secret="")


async def _fake_download_file(url: str, file_path: Path, headers: dict[str, str]):
    """Simulate download_file by creating a non-empty file and yielding packets."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"fake-adapter-data")
    yield PreDownloadPacket(17)
    yield DownloadedPacket(17)


@pytest.mark.asyncio
async def test_download_passes_url_through(downloader: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify the URL is passed to download_file as-is."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    captured_url = None

    async def capture_download(url: str, file_path: Path, headers: dict[str, str]):
        nonlocal captured_url
        captured_url = url
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"data")
        yield PreDownloadPacket(4)

    with patch("server.utils.model_downloader.download_file", side_effect=capture_download):
        async for _ in downloader.download("http://registry.local:5000/pirate-llama/adapter.gguf", model_dir, temp_dir):
            pass

    assert captured_url == "http://registry.local:5000/pirate-llama/adapter.gguf"


@pytest.mark.asyncio
async def test_download_yields_packets_and_success(downloader: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify the download yields PreDownloadPacket, DownloadedPacket, and SuccessDownloadPacket."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    with patch("server.utils.model_downloader.download_file", side_effect=_fake_download_file):
        packets = [p async for p in downloader.download("http://registry.local:5000/my-model/adapter.gguf", model_dir, temp_dir)]

    assert isinstance(packets[0], PreDownloadPacket)
    assert isinstance(packets[1], DownloadedPacket)
    assert isinstance(packets[-1], SuccessDownloadPacket)
    assert packets[-1].local_path == model_dir / "adapter.gguf"
    assert (model_dir / "adapter.gguf").exists()


@pytest.mark.asyncio
async def test_download_uses_custom_filename(downloader: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify the filename parameter overrides the default adapter.gguf."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    url = "http://registry.local:5000/my-model/adapter.gguf"
    with patch("server.utils.model_downloader.download_file", side_effect=_fake_download_file):
        packets = [p async for p in downloader.download(url, model_dir, temp_dir, "custom.gguf")]

    success = packets[-1]
    assert isinstance(success, SuccessDownloadPacket)
    assert success.local_path == model_dir / "custom.gguf"
    assert success.filename == "custom.gguf"


@pytest.mark.asyncio
async def test_download_skips_if_file_exists(downloader: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify download is skipped when the target file already exists."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "adapter.gguf").write_bytes(b"existing")
    temp_dir = tmp_path / "temp"

    mock_download = AsyncMock()
    with patch("server.utils.model_downloader.download_file", mock_download):
        packets = [p async for p in downloader.download("http://registry.local:5000/my-model/adapter.gguf", model_dir, temp_dir)]

    mock_download.assert_not_called()
    assert len(packets) == 1
    assert isinstance(packets[0], SuccessDownloadPacket)


@pytest.mark.asyncio
async def test_download_passes_bearer_headers(downloader: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify Bearer token headers are passed to download_file."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    captured_headers = None

    async def capture_headers(url: str, file_path: Path, headers: dict[str, str]):
        nonlocal captured_headers
        captured_headers = headers
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"data")
        yield PreDownloadPacket(4)

    with patch("server.utils.model_downloader.download_file", side_effect=capture_headers):
        async for _ in downloader.download("http://registry.local:5000/my-model/adapter.gguf", model_dir, temp_dir):
            pass

    assert captured_headers == {"Authorization": "Bearer test-secret"}


@pytest.mark.asyncio
async def test_download_no_secret_sends_empty_headers(downloader_no_secret: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify empty headers are sent when no secret is configured."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    captured_headers = None

    async def capture_headers(url: str, file_path: Path, headers: dict[str, str]):
        nonlocal captured_headers
        captured_headers = headers
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"data")
        yield PreDownloadPacket(4)

    with patch("server.utils.model_downloader.download_file", side_effect=capture_headers):
        async for _ in downloader_no_secret.download("http://registry.local:5000/my-model/adapter.gguf", model_dir, temp_dir):
            pass

    assert captured_headers == {}


@pytest.mark.asyncio
async def test_download_handles_401_error(downloader: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify HttpClientError with 401 is converted to HTTPException with friendly message."""
    from fastapi import HTTPException

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    async def fail_download(url: str, file_path: Path, headers: dict[str, str]):
        raise HttpClientError(message="401 Unauthorized", status_code=401, headers=CIMultiDictProxy(CIMultiDict()), body="401")
        yield  # make it an async generator

    with patch("server.utils.model_downloader.download_file", side_effect=fail_download), pytest.raises(HTTPException) as exc_info:
        async for _ in downloader.download("http://registry.local:5000/my-model/adapter.gguf", model_dir, temp_dir):
            pass

    assert exc_info.value.status_code == 500
    assert "Adapter registry authentication failed" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_download_cleans_temp_on_error(downloader: AdapterRegistryDownloader, tmp_path: Path) -> None:
    """Verify temp files are cleaned up when download fails."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    async def fail_after_write(url: str, file_path: Path, headers: dict[str, str]):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"partial")
        raise HttpClientError(message="500 Server Error", status_code=500, headers=CIMultiDictProxy(CIMultiDict()), body="error")
        yield

    from fastapi import HTTPException

    with patch("server.utils.model_downloader.download_file", side_effect=fail_after_write), pytest.raises(HTTPException):
        async for _ in downloader.download("http://registry.local:5000/my-model/adapter.gguf", model_dir, temp_dir):
            pass

    # Temp dir should have no leftover files
    if temp_dir.exists():
        assert list(temp_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_check_url(downloader: AdapterRegistryDownloader) -> None:
    """Verify check_url matches registry URLs."""
    assert downloader.check_url("http://registry.local:5000/my-model/adapter.gguf") is True
    assert downloader.check_url("http://registry.local:5000/") is True
    assert downloader.check_url("https://huggingface.co/model") is False
    assert downloader.check_url("http://other-host:5000/model") is False


@pytest.mark.asyncio
async def test_check_url_localhost_normalization() -> None:
    """Verify check_url treats localhost and 127.0.0.1 as equivalent."""
    dl_localhost = AdapterRegistryDownloader(url="http://localhost:8333", secret="s")
    assert dl_localhost.check_url("http://127.0.0.1:8333/pirate-llama") is True

    dl_ip = AdapterRegistryDownloader(url="http://127.0.0.1:8333", secret="s")
    assert dl_ip.check_url("http://localhost:8333/pirate-llama") is True


@pytest.mark.asyncio
async def test_download_localhost_normalization(tmp_path: Path) -> None:
    """Verify download works when registry uses 127.0.0.1 but Modelfile URL uses localhost."""
    dl = AdapterRegistryDownloader(url="http://127.0.0.1:8333", secret="s")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    temp_dir = tmp_path / "temp"

    captured_url = None

    async def capture_download(url: str, file_path: Path, headers: dict[str, str]):
        nonlocal captured_url
        captured_url = url
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"data")
        yield PreDownloadPacket(4)

    with patch("server.utils.model_downloader.download_file", side_effect=capture_download):
        async for _ in dl.download("http://localhost:8333/pirate-llama/adapter.gguf", model_dir, temp_dir):
            pass

    # URL is passed through as-is (localhost is fine — it's the same host)
    assert captured_url == "http://localhost:8333/pirate-llama/adapter.gguf"
