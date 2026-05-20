# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for copy_envs.py script."""

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.copy_envs import main, replace_hosts


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        ("MONGO_URL=mongo:27017\n", "MONGO_URL=localhost:27017\n"),
        ("INFLUX_URL=http://infra:8086\n", "INFLUX_URL=http://localhost:8086\n"),
        ("API=https://something:1234\n", "API=https://localhost:1234\n"),
        ("REDIS=redis://cache:6379\n", "REDIS=redis://localhost:6379\n"),
        ("DB=postgres://db:5432\n", "DB=postgres://localhost:5432\n"),
        ("# just a comment\n", "# just a comment\n"),
        ("URL=http://localhost:8080\n", "URL=http://localhost:8080\n"),
        ("", ""),
        ("SECRET_KEY=abc123\n", "SECRET_KEY=abc123\n"),
        ("A=http://host1:80 B=host2:90\n", "A=http://localhost:80 B=localhost:90\n"),
    ],
    ids=[
        "plain_host_port",
        "http_url",
        "https_url",
        "redis_scheme",
        "postgres_scheme",
        "comment",
        "already_localhost",
        "empty_line",
        "no_host",
        "multiple_hosts",
    ],
)
def test_replace_hosts(input_str: str, expected: str):
    assert replace_hosts(input_str) == expected


def test_main_raises_when_src_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent" / ".env"
    with patch("scripts.copy_envs.SRC_ENV", missing), pytest.raises(FileNotFoundError, match=r"Source \.env not found"):
        main()


def test_main_copies_and_replaces(tmp_path: Path) -> None:
    src = tmp_path / "src.env"
    dst = tmp_path / "dst.env"
    src.write_text("REDIS=redis://cache:6379\nSECRET=x\n", encoding="utf-8")

    with patch("scripts.copy_envs.SRC_ENV", src), patch("scripts.copy_envs.DEST_ENV", dst):
        main()

    content = dst.read_text(encoding="utf-8")
    assert "localhost:6379" in content
    assert "SECRET=x" in content


def test_main_dest_file_created(tmp_path: Path) -> None:
    src = tmp_path / "src.env"
    dst = tmp_path / "dst.env"
    src.write_text("K=v\n", encoding="utf-8")

    with patch("scripts.copy_envs.SRC_ENV", src), patch("scripts.copy_envs.DEST_ENV", dst):
        main()

    assert dst.exists()


def test_main_empty_src(tmp_path: Path) -> None:
    src = tmp_path / "src.env"
    dst = tmp_path / "dst.env"
    src.write_text("", encoding="utf-8")

    with patch("scripts.copy_envs.SRC_ENV", src), patch("scripts.copy_envs.DEST_ENV", dst):
        main()

    assert dst.read_text(encoding="utf-8") == ""
