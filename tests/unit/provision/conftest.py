# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Make the chart's provisioning package importable the way the Job imports it.

The package lives in the chart (deploy/helm/deepfellow/files/deepfellow_provision), which mounts
it into the Job's pod and runs `python -m deepfellow_provision`; it is not installed into the venv.
"""

import sys
from pathlib import Path

FILES_DIR = Path(__file__).resolve().parents[3] / "deploy" / "helm" / "deepfellow" / "files"
if str(FILES_DIR) not in sys.path:
    sys.path.insert(0, str(FILES_DIR))
