# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Claude service."""

from server.models.services import ServiceField, ServiceSpecification
from server.services.remote_service import BaseServiceOptions, RemoteConst, RemoteModel, RemoteService

_const = RemoteConst(
    models={
        # Claude 4.6 - 1M with beta header
        "claude-opus-4-6": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=1_048_576,
            legacy_completions=False,
            responses=False,
        ),
        "claude-sonnet-4-6": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=1_048_576,
            legacy_completions=False,
            responses=False,
        ),
        # Claude 4.5 family
        "claude-haiku-4-5": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            legacy_completions=False,
            responses=False,
        ),
        "claude-haiku-4-5-20251001": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            legacy_completions=False,
            responses=False,
        ),
        "claude-opus-4-5": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=1_048_576,  # 1M with beta header
            legacy_completions=False,
            responses=False,
        ),
        "claude-sonnet-4-5": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=1_048_576,  # 1M with beta header
            legacy_completions=False,
            responses=False,
        ),
        # Claude 4.1 / 4.0 - 200K only
        "claude-opus-4-1": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            legacy_completions=False,
            responses=False,
        ),
        "claude-opus-4-0": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            legacy_completions=False,
            responses=False,
        ),
        "claude-sonnet-4-0": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            legacy_completions=False,
            responses=False,
        ),
    }
)


class ClaudeServiceOptions(BaseServiceOptions):
    """Claude authorization header included."""

    api_key: str
    anthropic_version: str
    anthropic_beta: str = ""

    @property
    def headers(self) -> dict[str, str]:
        """Translate data into header."""
        response = {"x-api-key": f"{self.api_key}", "anthropic-version": self.anthropic_version}
        if self.anthropic_beta:
            response["anthropic-beta"] = self.anthropic_beta

        return response


class ClaudeService(RemoteService[ClaudeServiceOptions]):
    options_class = ClaudeServiceOptions

    def get_type(self) -> str:
        """Return the service id."""
        return "claude"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to Claude models."

    def get_default_url(self) -> str:
        """Return the default url."""
        return "https://api.anthropic.com"

    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""
        return _const

    def get_spec(self) -> ServiceSpecification:
        """Provide the specification for the install service modal fields compatible with Claude."""
        return ServiceSpecification(
            fields=[
                ServiceField(
                    type="text",
                    name="api_url",
                    description="API URL",
                    required=False,
                    default=self.get_default_url(),
                ),
                ServiceField(
                    type="password",
                    name="api_key",
                    description="API Key",
                    required=True,
                ),
                ServiceField(
                    type="text",
                    name="anthropic_version",
                    description="Version of the Anthropic API",
                    default="2023-06-01",
                    required=True,
                ),
                ServiceField(
                    type="text",
                    name="anthropic_beta",
                    description="Beta header. Set 'context-1m-2025-08-07' to allow 1M context on Opus and Sonnet",
                    placeholder="context-1m-2025-08-07",
                    required=False,
                ),
            ]
        )
