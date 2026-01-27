# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""GoogleAI service."""

from server.services.remote_service import RemoteConst, RemoteModel, RemoteService

_const = RemoteConst(
    models={
        "gemini-1.5-pro-latest": RemoteModel(type="llm", responses=False),
        "gemini-1.5-pro-002": RemoteModel(type="llm", responses=False),
        "gemini-1.5-pro": RemoteModel(type="llm", responses=False),
        "gemini-1.5-flash-latest": RemoteModel(type="llm", responses=False),
        "gemini-1.5-flash": RemoteModel(type="llm", responses=False),
        "gemini-1.5-flash-002": RemoteModel(type="llm", responses=False),
        "gemini-1.5-flash-8b": RemoteModel(type="llm", responses=False),
        "gemini-1.5-flash-8b-001": RemoteModel(type="llm", responses=False),
        "gemini-1.5-flash-8b-latest": RemoteModel(type="llm", responses=False),
        "gemini-2.5-pro-preview-03-25": RemoteModel(type="llm", responses=False),
        "gemini-2.5-flash-preview-05-20": RemoteModel(type="llm", responses=False),
        "gemini-2.5-flash": RemoteModel(type="llm", responses=False),
        "gemini-2.5-flash-lite-preview-06-17": RemoteModel(type="llm", responses=False),
        "gemini-2.5-pro-preview-05-06": RemoteModel(type="llm", responses=False),
        "gemini-2.5-pro-preview-06-05": RemoteModel(type="llm", responses=False),
        "gemini-2.5-pro": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-exp": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-001": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-lite-001": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-lite": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-lite-preview-02-05": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-lite-preview": RemoteModel(type="llm", responses=False),
        "gemini-2.0-pro-exp": RemoteModel(type="llm", responses=False),
        "gemini-2.0-pro-exp-02-05": RemoteModel(type="llm", responses=False),
        "gemini-exp-1206": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-thinking-exp-01-21": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-thinking-exp": RemoteModel(type="llm", responses=False),
        "gemini-2.0-flash-thinking-exp-1219": RemoteModel(type="llm", responses=False),
        "learnlm-2.0-flash-experimental": RemoteModel(type="llm", responses=False),
        "gemma-3-1b-it": RemoteModel(type="llm", responses=False),
        "gemma-3-4b-it": RemoteModel(type="llm", responses=False),
        "gemma-3-12b-it": RemoteModel(type="llm", responses=False),
        "gemma-3-27b-it": RemoteModel(type="llm", responses=False),
        "gemma-3n-e4b-it": RemoteModel(type="llm", responses=False),
        "gemma-3n-e2b-it": RemoteModel(type="llm", responses=False),
        "gemini-2.5-flash-lite": RemoteModel(type="llm", responses=False),
        "gemini-2.5-flash-image-preview": RemoteModel(type="txt2img"),  # NOTE: aka Nano Banana
        "embedding-001": RemoteModel(type="embedding"),
        "text-embedding-004": RemoteModel(type="embedding"),
        "gemini-embedding-exp-03-07": RemoteModel(type="embedding"),
        "gemini-embedding-exp": RemoteModel(type="embedding"),
        "gemini-embedding-001": RemoteModel(type="embedding"),
        "imagen-3.0-generate-002": RemoteModel(type="txt2img"),
        "imagen-4.0-generate-preview-06-06": RemoteModel(type="txt2img"),
        "imagen-4.0-ultra-generate-preview-06-06": RemoteModel(type="txt2img"),
        # TTS Models are not open https://cloud.google.com/text-to-speech/docs/gemini-tts#curl ; speech is not listed here: https://ai.google.dev/gemini-api/docs/openai
        # "gemini-2.5-flash-preview-tts": RemoteModel(type="tts"),
        # "gemini-2.5-pro-preview-tts": RemoteModel(type="tts"),
    }
)


class GoogleAIService(RemoteService):
    api_version = "v1beta/openai/"

    def get_id(self) -> str:
        """Return the service id."""
        return "google"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to Google AI models."

    def get_default_url(self) -> str:
        """Return the default url."""
        return "https://generativelanguage.googleapis.com"

    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""
        return _const
