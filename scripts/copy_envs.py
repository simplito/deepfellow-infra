# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
# 
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
# 
# See the License for the specific language governing permissions and
# limitations under the License.

# Development script only
# Used to fast copy paste .env from infra created by DeepFellow CLI, and replace docker hosts with localhost.
# Allows to connect local deepfellow infra to docker subservices.

#!/usr/bin/env python3

import re
from pathlib import Path

# ruff: noqa: T201

# Source and destination paths
HOME = Path.home()
SRC_ENV = HOME / ".deepfellow" / "infra" / ".env"

# Parent directory of the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent
DEST_ENV = SCRIPT_DIR.parent / ".env"

# Regex patterns to replace hostnames with localhost
# Handles:
#   mongo:27017
#   http://infra:8086
#   https://something:1234
#   redis://cache:6379
HOST_REPLACEMENTS = [
    # http(s)://host:port
    (re.compile(r"(https?://)([^:/\s]+)(:\d+)"), r"\1localhost\3"),
    # scheme://host:port (e.g., redis://, postgres://)
    (re.compile(r"([a-zA-Z]+://)([^:/\s]+)(:\d+)"), r"\1localhost\3"),
    # plain host:port (no protocol)
    (re.compile(r"(?<!://)(\b[^=\s:/]+)(:\d+)"), r"localhost\2"),
]


def replace_hosts(line: str) -> str:
    """Replace docker hosts to localhost."""
    for pattern, repl in HOST_REPLACEMENTS:
        line = pattern.sub(repl, line)
    return line


def main() -> None:
    """Copy paste .env from infra created by Deepfellow CLI, and replace docker url with localhost."""
    if not SRC_ENV.exists():
        msg = f"Source .env not found: {SRC_ENV}"
        raise FileNotFoundError(msg)

    with SRC_ENV.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = [replace_hosts(line) for line in lines]

    DEST_ENV.write_text("".join(new_lines), encoding="utf-8")
    print(f"Copied and rewritten {SRC_ENV} -> {DEST_ENV} with localhost hosts.")


if __name__ == "__main__":
    main()
