# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom loggers."""

import logging

# uvicorn_logger is created to print out the logs in the terminal
# when running the server with `uvicorn server.main:app ...`
uvicorn_logger = logging.getLogger("uvicorn")


# TODO Configure to log to the DeepFellow log collector
def get_logger(module: str) -> logging.Logger:
    """Get a logger for a module.

    This logger is the default logger of the DeepFellow Server.
    """
    return logging.getLogger(module)
