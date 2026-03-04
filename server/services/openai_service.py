# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenAI service."""

from server.services.remote_service import DefaultRemoteServiceOptions, RemoteConst, RemoteModel, RemoteService

_const = RemoteConst(
    models={
        "davinci-002": RemoteModel(
            type="llm", context_length=16_385, max_context_length=16_385, completions=False, responses=False, messages=False
        ),
        "babbage-002": RemoteModel(
            type="llm", context_length=16_385, max_context_length=16_385, completions=False, responses=False, messages=False
        ),
        "gpt-3.5-turbo": RemoteModel(type="llm", context_length=16_385, max_context_length=16_385, messages=False),
        "gpt-3.5-turbo-instruct": RemoteModel(
            type="llm", context_length=4_096, max_context_length=4_096, completions=False, responses=False, messages=False
        ),
        "gpt-3.5-turbo-16k": RemoteModel(
            type="llm", context_length=16_385, max_context_length=16_385, legacy_completions=False, responses=False, messages=False
        ),
        "gpt-4-turbo": RemoteModel(type="llm", context_length=128_000, max_context_length=128_000, messages=False),
        "gpt-4.1": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gpt-4.1-mini": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gpt-4.1-nano": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gpt-4o": RemoteModel(type="llm", context_length=128_000, max_context_length=128_000, messages=False),
        "gpt-4o-mini": RemoteModel(type="llm", context_length=128_000, max_context_length=128_000, messages=False),
        "gpt-4o-transcribe": RemoteModel(type="stt"),
        "gpt-4o-mini-transcribe": RemoteModel(type="stt"),
        "gpt-4o-mini-tts": RemoteModel(type="tts"),
        "gpt-4": RemoteModel(type="llm", context_length=8_196, max_context_length=8_196, messages=False),
        "gpt-5": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5-mini": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5-nano": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.1": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.1-codex": RemoteModel(
            type="llm",
            context_length=400_000,
            max_context_length=400_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "gpt-5.1-codex-mini": RemoteModel(
            type="llm",
            context_length=400_000,
            max_context_length=400_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "gpt-5.2": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.2-pro": RemoteModel(
            type="llm",
            context_length=400_000,
            max_context_length=400_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "o1": RemoteModel(type="llm", messages=False),
        "o1-pro": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "o3": RemoteModel(type="llm", context_length=200_000, max_context_length=200_000, messages=False),
        "o3-mini": RemoteModel(type="llm", context_length=200_000, max_context_length=200_000, messages=False),
        "o4-mini": RemoteModel(type="llm", context_length=200_000, max_context_length=200_000, messages=False),
        "o4-mini-deep-research": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
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
    options_class = DefaultRemoteServiceOptions

    def get_type(self) -> str:
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
