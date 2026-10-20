# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Progress output. The Job's log is the operator's only view of a run."""

import logging

logger = logging.getLogger("deepfellow_provision")


def configure() -> None:
    """Send `[provision] <message>` lines to stdout."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[provision] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
