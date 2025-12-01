# DeepFellow Software Framework.
# Copyright © 2025 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for chat completions."""

from abc import abstractmethod
from typing import Annotated, Any, Literal

from aiohttp import FormData
from fastapi import File as FastApiFile
from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator


class ImageUrl(BaseModel):
    url: str
    detail: str = "auto"


class InputAudio(BaseModel):
    data: str
    format: str


class File(BaseModel):
    file_data: str = ""
    file_id: str = ""
    filename: str = ""


class TextContentPart(BaseModel):
    text: str
    type: str = "text"


class ImageContentPart(BaseModel):
    image_url: ImageUrl
    type: str


class AudioContentPart(BaseModel):
    input_audio: InputAudio
    type: str


class FileContentPart(BaseModel):
    file: File
    type: str


ContentPart = TextContentPart | ImageContentPart | AudioContentPart | FileContentPart


class BaseMessage(BaseModel):
    content: str | list[ContentPart] = []
    role: Literal["user", "system", "developer", "assistant"]
    name: str | None = None


class DeveloperMessage(BaseModel):
    content: str | list[ContentPart] = []
    role: Literal["developer"] = "developer"
    name: str | None = None


class SystemMessage(BaseModel):
    content: str | list[ContentPart] = []
    role: Literal["system"] = "system"
    name: str | None = None


class UserMessage(BaseModel):
    content: str | list[ContentPart] = []
    role: Literal["user"] = "user"
    name: str | None = None


class ToolCallFunction(BaseModel):
    arguments: str
    name: str


class ToolCall(BaseModel):
    function: ToolCallFunction
    id: str
    type: Literal["function"] = "function"


class AssistantMessage(BaseModel):
    content: str | list[str] | None = None
    role: Literal["assistant"] = "assistant"
    name: str | None = None
    audio: str | None = None
    refusal: str | None = None
    tool_calls: list[ToolCall] = []


class ToolMessage(BaseModel):
    content: str | list[str] = []
    role: Literal["tool"] = "tool"
    tool_call_id: str


ChatMessage = DeveloperMessage | SystemMessage | UserMessage | BaseMessage | AssistantMessage | ToolMessage


class ToolFunction(BaseModel):
    name: Annotated[str, Field(description="The name of the function to be called")]
    description: Annotated[str, Field(description="A description of what the function does")] = ""
    parameters: Annotated[dict[str, Any], Field(description="The parameters the function accepts")]
    strict: bool = False


class ToolChat(BaseModel):
    type: Literal["function"] = "function"
    function: Annotated[ToolFunction, Field(description="The function definition")]


class ResponseFormat(BaseModel):
    type: Annotated[str, Field(description="The format type, either 'text' or 'json_object'")] = "text"


class ChatCompletionRequest(BaseModel):
    messages: Annotated[
        list[ChatMessage],
        Field(
            description=(
                "A list of messages comprising the conversation so far. Depending on the model you use, "
                "different message types (modalities) are supported, like text, images, and audio."
            )
        ),
    ]
    model: Annotated[
        str,
        Field(
            description=(
                "Model ID used to generate the response, like llama3 or deepseek-r1. "
                "Deepfellow supports a wide range of models with different capabilities, and performance "
                "characteristics."
            )
        ),
    ]
    tools: Annotated[
        list[ToolChat] | None,
        Field(
            description=(
                "A list of tools the model may call. Currently, only functions are supported as a tool. Use this to "
                "provide a list of functions the model may generate JSON inputs for."
            )
        ),
    ] = None
    tool_choice: Annotated[
        str | dict[str, Any] | None,
        Field(
            description=(
                "Controls which (if any) tool is called by the model. `null` means the model will not call any tool "
                "and instead generates a message. `auto` means the model can pick between generating a message or calling "
                "one or more tools. `required` means the model must call one or more tools. Specifying a particular tool "
                'via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.'
                "`none` is the default when no tools are present. `auto` is the default if tools are present."
            )
        ),
    ] = None
    temperature: Annotated[
        float | None,
        Field(
            description=(
                "What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more "
                "random, while lower values like `0.2` will make it more focused and deterministic. We generally recommend "
                "altering this or `top_p` but not both."
            )
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            description=(
                "An alternative to sampling with temperature, called nucleus sampling, where the model considers the "
                "results of the tokens with `top_p` probability mass. So `0.1` means only the tokens comprising the top "
                "10% probability mass are considered. We generally recommend altering this or `temperature` but not both."
            )
        ),
    ] = None
    n: Annotated[
        int | None,
        Field(
            description=(
                "How many chat completion choices to generate for each input message."
                "Keep `n` as `1` to receive one choice per input message."
            )
        ),
    ] = None
    stream: Annotated[
        bool | None,
        Field(
            description=(
                "If set to `true`, the model response data will be streamed to the client as it is generated using server-sent events."
            )
        ),
    ] = None
    stream_options: "StreamOptions | None" = None
    max_completion_tokens: Annotated[
        int | None,
        Field(
            description=(
                "An upper bound for the number of tokens that can be generated for a completion, including visible "
                "output tokens and reasoning tokens."
            )
        ),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(
            description=(
                "The maximum number of tokens that can be generated in the chat completion. This value can be "
                "used to control costs for text generated via API. This value is now deprecated in favor of "
                "`max_completion_tokens`, and is not compatible with o-series models."
            ),
            deprecated=True,
        ),
    ] = None
    response_format: Annotated[
        ResponseFormat | None,
        Field(
            description=(
                "An object specifying the format that the model must output. Setting to "
                '`{"type": "json_schema", "json_schema": {...}}` enables Structured Outputs which ensures the model will '
                'match your supplied JSON schema. Setting to `{"type": "json_object"}` enables the older JSON mode, '
                "which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for "
                "models that support it."
            )
        ),
    ] = None
    seed: Annotated[
        int | None,
        Field(
            description=(
                "If specified, our system will make a best effort to sample deterministically, such that repeated "
                "requests with the same seed and parameters should return the same result. Determinism is not guaranteed, "
                "and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend."
            )
        ),
    ] = None
    stop: Annotated[
        str | list[str] | None,
        Field(
            description=(
                "Might not be supported with latest reasoning models like `o3` and `o4-mini`. Up to 4 sequences where "
                "the API will stop generating further tokens. The returned text will not contain the stop sequence."
            )
        ),
    ] = None
    user: Annotated[
        str | None,
        Field(description=("A stable identifier for your end-users. Used to boost cache hit rates by better bucketing similar requests.")),
    ] = None
    safety_identifier: Annotated[
        str | None,
        Field(
            description=(
                "A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies. "
                "The IDs should be a string that uniquely identifies each user. We recommend hashing their username or email address, "
                "in order to avoid sending us any identifying information."
            )
        ),
    ] = None
    prompt_cache_key: Annotated[
        str | None,
        Field(
            description=(
                "Used by OpenAI to cache responses for similar requests to optimize your cache hit rates. Replaces the user field."
            )
        ),
    ] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "llama3",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "temperature": 0.7,
                    "max_completion_tokens": 500,
                    "stream": False,
                }
            ]
        }
    }


class EmbeddingRequest(BaseModel):
    input: Annotated[str | list[str], Field(description="The input text(s) to embed")]
    model: Annotated[str, Field(description="ID of the model to use for embedding", examples=["llama3"])]
    dimensions: Annotated[int | None, Field(description="The number of dimensions to return for each embedding")] = None
    encoding_format: Annotated[str | None, Field(description="The encoding format of the embeddings", examples=["float"])] = None
    user: Annotated[str | None, Field(description="A unique identifier for the end-user", examples=["user123"])] = None

    model_config = {"json_schema_extra": {"examples": [{"input": "Hello, how are you?", "model": "llama3", "encoding_format": "float"}]}}


class ImagesRequest(BaseModel):
    prompt: str
    background: Literal["auto", "transparent", "opaque"] | None = None
    model: str
    moderation: str | None = None
    n: Annotated[int | None, Field(ge=1, le=4)] = None
    output_compression: Annotated[int | None, Field(ge=0, le=100)] = 95
    output_format: Literal["png", "webp", "jpeg"] | None = None
    quality: Literal["low", "medium", "high", "auto"] | None = None
    response_format: Literal["url", "b64_json"] | None = None
    size: str | None = None
    style: str | None = None
    user: str | None = None
    partial_images: int | None = None
    stream: bool | None = None


class CreateSpeechRequest(BaseModel):
    input: Annotated[str, Field(max_length=4096, description="The text to generate audio for. The maximum length is 4096 characters.")]
    model: Annotated[str, Field(description="One of the available TTS models.")]
    voice: Annotated[str | None, Field(description="The voice to use when generating the audio.")] = None
    instructions: Annotated[str | None, Field(description="Control the voice of your generated audio with additional instructions.")] = None
    format: Annotated[
        Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None,
        Field(description="The format to audio in. Supported formats are mp3, opus, aac, flac, wav, and pcm."),
    ] = None
    speed: Annotated[
        float | None,
        Field(ge=0.25, le=4.0, description="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default."),
    ] = None
    stream_format: Annotated[
        Literal["sse", "audio"] | None,
        Field(
            description=(
                "The format to stream the audio in. Supported formats are sse and audio. sse is not supported for tts-1 or tts-1-hd."
            )
        ),
    ] = None


class ModelProps(BaseModel):
    private: bool


ModelType = Literal["tts", "stt", "txt2img", "embedding", "llm", "llm-only-v1", "llm-only-v2", "custom"]

type ModelId = str
type RegistrationId = str


class Model(BaseModel):
    id: RegistrationId
    name: ModelId
    type: ModelType
    props: ModelProps
    usage: int


class ApiModel(BaseModel):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    props: ModelProps


class ApiModels(BaseModel):
    object: Literal["list"] = "list"
    data: list[ApiModel]


type TranscriptionInclude = Literal["logprobs"]

languages = [
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
    "yue",
]


class AudioChunkingStrategy(BaseModel):
    type: Annotated[Literal["server_vad"], Field(description="Must be set to 'server_vad' to enable manual chunking using server side VAD")]

    prefix_padding_ms: Annotated[
        int | None,
        Field(
            ge=0,
            description=(
                "Amount of audio to include before the VAD detected speech (in milliseconds). "
                "This ensures the beginning of speech is not cut off."
            ),
        ),
    ] = None

    silence_duration_ms: Annotated[
        int | None,
        Field(
            ge=0,
            description=(
                "Duration of silence to detect speech stop (in milliseconds). "
                "With shorter values the model will respond more quickly, "
                "but may jump in on short pauses from the user."
            ),
        ),
    ] = None

    threshold: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "Sensitivity threshold (0.0 to 1.0) for voice activity detection. "
                "A higher threshold will require louder audio to activate the model, "
                "and thus might perform better in noisy environments."
            ),
        ),
    ] = None


class FormSerializable(BaseModel):
    @abstractmethod
    async def to_form(self, remove_model: bool, rewrite_model_to: str | None) -> FormData:
        """Serialize to FormData."""


class CreateTranscriptionRequest(FormSerializable):
    model: Annotated[str, Field(description="ID of the model to use.")]
    file: UploadFile = FastApiFile(
        description="The audio file object (not file name) in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.",
    )
    temperature: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, "
                "while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability "
                "to automatically increase the temperature until certain thresholds are hit."
            ),
        ),
    ] = None
    type: Annotated[str | None, Field(description="The format type, either 'text' or 'json_object'")] = None
    chunking_strategy: Annotated[
        Literal["auto"] | AudioChunkingStrategy | None,
        Field(
            description=(
                "Controls how the audio is cut into chunks."
                'When set to "auto", the server first normalizes loudness and'
                " then uses voice activity detection (VAD) to choose boundaries."
                " server_vad object can be provided to tweak VAD detection parameters manually."
                " If unset, the audio is transcribed as a single block."
            )
        ),
    ] = None
    include: Annotated[
        list[TranscriptionInclude] | None,
        Field(
            alias="include[]",
            description=(
                "Additional information to include in the transcription response. "
                "logprobs will return the log probabilities of the tokens in the response to "
                "understand the model's confidence in the transcription."
            ),
        ),
    ] = None
    language: Annotated[
        str | None,
        Field(
            description=(
                "The language of the input audio. Supplying the input language in ISO-639-1 (e.g. en) "
                "format will improve accuracy and latency."
            )
        ),
    ] = None
    prompt: Annotated[
        str | None,
        Field(
            description=(
                "An optional text to guide the model's style or continue a previous audio segment. "
                "The prompt should match the audio language."
            )
        ),
    ] = None
    stream: Annotated[
        bool | None,
        Field(
            description=(
                "If set to true, the model response data will be streamed to the client as it is generated using server-sent events."
            )
        ),
    ] = None
    timestamp_granularities: Annotated[
        list[Literal["word", "segment"]] | None,
        Field(
            alias="timestamp_granularities[]",
            description=(
                "The timestamp granularities to populate for this transcription. response_format must be set verbose_json to use "
                "timestamp granularities. "
                "Either or both of these options are supported: word, or segment."
                "Note: There is no additional latency for segment timestamps, "
                "but generating word timestamps incurs additional latency."
            ),
        ),
    ] = None

    @field_validator("language", mode="after")
    def _check_language(cls, v: str | None) -> str | None:  # noqa: N805
        if v is not None and v not in languages:
            msg = f"Unsupported value, available values: {', '.join(languages)}"
            raise ValueError(msg)
        return v

    async def to_form(self, remove_model: bool, rewrite_model_to: str | None) -> FormData:
        """Serialize to FormData."""
        form = FormData()

        for field_name in CreateTranscriptionRequest.model_fields:
            if field_name == "file":
                form.add_field("file", await self.file.read(), filename=self.file.filename, content_type=self.file.content_type)  # type: ignore
            elif field_name == "model":
                if not remove_model:
                    form.add_field("model", rewrite_model_to if rewrite_model_to else self.model)
            else:
                field_value = getattr(self, field_name)
                if field_value:
                    if isinstance(field_value, list):
                        for element in field_value:  # pyright: ignore[reportUnknownVariableType]
                            form.add_field(field_name + "[]", element)
                    elif isinstance(field_value, bool):
                        form.add_field(field_name, "true" if field_value else "false")
                    elif isinstance(field_value, float):
                        form.add_field(field_name, str(field_value))
                    else:
                        form.add_field(field_name, field_value)
        return form


type Bias = Annotated[int, Field(ge=-100, le=100)]
type StopSequenceList = Annotated[list[str], Field(max_length=4)]


class StreamOptions(BaseModel):
    """Options for streaming response."""

    include_obfuscation: Annotated[
        bool | None, Field(description="Enable stream obfuscation to normalize payload sizes as mitigation for side-channel attacks.")
    ] = None
    include_usage: Annotated[bool | None, Field(description="Include token usage statistics in the stream.")] = None


class CompletionLegacyRequest(BaseModel):
    model: Annotated[str, Field(description="ID of the model to use.")]
    prompt: Annotated[
        str | list[str] | list[int] | list[list[int]],
        Field(description="The prompt(s) to generate completions for - can be string, array of strings, tokens, or token arrays."),
    ]
    best_of: Annotated[
        int | None, Field(ge=1, description="Generate best_of completions and return the best one. Must be greater than n if both are set.")
    ] = None
    echo: Annotated[bool | None, Field(description="Echo back the prompt in addition to the completion.")] = None
    frequency_penalty: Annotated[
        float | None, Field(ge=-2.0, le=2.0, description="Penalize new tokens based on their existing frequency (-2.0 to 2.0).")
    ] = None
    logit_bias: Annotated[
        dict[str, Bias] | None, Field(description="Modify likelihood of specified tokens appearing (token ID -> bias from -100 to 100).")
    ] = None
    logprobs: Annotated[int | None, Field(ge=0, le=5, description="Include log probabilities on the most likely tokens (max 5).")] = None
    max_tokens: Annotated[int | None, Field(ge=1, description="Maximum number of tokens to generate in the completion.")] = None
    n: Annotated[int | None, Field(ge=1, description="How many completions to generate for each prompt.")] = None
    presence_penalty: Annotated[
        float | None,
        Field(ge=-2.0, le=2.0, description="Penalize new tokens based on whether they appear in the text so far (-2.0 to 2.0)."),
    ] = None
    seed: Annotated[int | None, Field(description="Seed for deterministic sampling (best effort, not guaranteed).")] = None
    stop: Annotated[str | StopSequenceList | None, Field(description="Up to 4 sequences where the API will stop generating.")] = None
    stream: Annotated[bool | None, Field(description="Stream back partial progress as server-sent events.")] = None
    stream_options: Annotated[StreamOptions | None, Field(description="Options for streaming response (only used when stream=True).")] = (
        None
    )
    suffix: Annotated[str | None, Field(description="Suffix after completion of inserted text.")] = None
    temperature: Annotated[
        float | None, Field(ge=0.0, le=2.0, description="Sampling temperature (0-2). Higher = more random, lower = more deterministic.")
    ] = None
    top_p: Annotated[
        float | None, Field(ge=0.0, le=1.0, description="Nucleus sampling - consider tokens with top_p probability mass (0-1).")
    ] = None
    user: Annotated[str | None, Field(description="Unique identifier for end-user to help monitor and detect abuse.")] = None
