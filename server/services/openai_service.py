# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI service."""

from server.services.remote_service import RemoteConst, RemoteModel, RemoteService

_const = RemoteConst(
    models={
        "davinci-002": RemoteModel(type="llm", completions=False),
        "babbage-002": RemoteModel(type="llm", completions=False),
        "gpt-3.5-turbo": RemoteModel(type="llm"),
        "gpt-3.5-turbo-instruct": RemoteModel(type="llm", completions=False),
        "gpt-3.5-turbo-16k": RemoteModel(type="llm"),
        "gpt-4-turbo": RemoteModel(type="llm"),
        "gpt-4.1": RemoteModel(type="llm"),
        "gpt-4.1-mini": RemoteModel(type="llm"),
        "gpt-4.1-nano": RemoteModel(type="llm"),
        "gpt-4o": RemoteModel(type="llm"),
        "gpt-4o-mini": RemoteModel(type="llm"),
        "gpt-4o-transcribe": RemoteModel(type="stt"),
        "gpt-4o-mini-transcribe": RemoteModel(type="stt"),
        "gpt-4o-mini-tts": RemoteModel(type="tts"),
        "gpt-4": RemoteModel(type="llm"),
        "gpt-5": RemoteModel(type="llm"),
        "gpt-5-mini": RemoteModel(type="llm"),
        "gpt-5-nano": RemoteModel(type="llm"),
        "gpt-5.1": RemoteModel(type="llm"),
        # "gpt-5.1-codex": RemoteModel(type="llm"), only /v1/responses
        # "gpt-5.1-codex-mini": RemoteModel(type="llm"), only /v1/responses
        "gpt-5.2": RemoteModel(type="llm"),
        # "gpt-5.2-pro": RemoteModel(type="llm"), only /v1/responses
        "o1": RemoteModel(type="llm"),
        "o1-mini": RemoteModel(type="llm"),
        # "o1-pro": RemoteModel(type="llm"), only /v1/responses
        "o3": RemoteModel(type="llm"),
        "o3-mini": RemoteModel(type="llm"),
        "o4-mini": RemoteModel(type="llm"),
        # "o4-mini-deep-research": RemoteModel(type="llm"), only /v1/responses
        "text-embedding-ada-002": RemoteModel(type="embedding"),
        "text-embedding-3-small": RemoteModel(type="embedding"),
        "text-embedding-3-large": RemoteModel(type="embedding"),
        "gpt-image-1": RemoteModel(type="txt2img"),
        "dall-e-2": RemoteModel(type="txt2img"),
        "dall-e-3": RemoteModel(type="txt2img"),
        "tts-1": RemoteModel(type="tts"),
        "tts-1-hd": RemoteModel(type="tts"),
        "whisper-1": RemoteModel(type="stt"),
    }
)


class OpenAIService(RemoteService):
    def get_id(self) -> str:
        """Return the service id."""
        return "openai"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to OpenAI models."

    def get_default_url(self) -> str:
        """Return the default url."""
        return "https://api.openai.com"

    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""
        return _const
