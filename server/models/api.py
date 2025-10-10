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
    name: str = Field(..., description="The name of the function to be called")
    description: str = Field("", description="A description of what the function does")
    parameters: dict[str, Any] = Field(..., description="The parameters the function accepts")
    strict: bool = False


class ToolChat(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction = Field(..., description="The function definition")


class ResponseFormat(BaseModel):
    type: str = Field("text", description="The format type, either 'text' or 'json_object'")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        description="A list of messages comprising the conversation so far. Depending on the model you use,"
        "different message types (modalities) are supported, like text, images, and audio.",
    )
    model: str = Field(
        ...,
        description="Model ID used to generate the response, like llama3 or deepseek-r1. "
        "Deepfellow supports a wide range of models with different capabilities, and performance "
        "characteristics.",
    )
    tools: list[ToolChat] | None = Field(
        None,
        description=(
            "A list of tools the model may call. Currently, only functions are supported as a tool. Use this to "
            "provide a list of functions the model may generate JSON inputs for."
        ),
    )
    tool_choice: str | dict[str, Any] | None = Field(
        None,
        description=(
            "Controls which (if any) tool is called by the model. `null` means the model will not call any tool "
            "and instead generates a message. `auto` means the model can pick between generating a message or calling "
            "one or more tools. `required` means the model must call one or more tools. Specifying a particular tool "
            'via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.\n\n'
            "`none` is the default when no tools are present. `auto` is the default if tools are present."
        ),
    )
    temperature: float | None = Field(
        None,
        description=(
            "What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more "
            "random, while lower values like `0.2` will make it more focused and deterministic. We generally recommend "
            "altering this or `top_p` but not both."
        ),
    )
    top_p: float | None = Field(
        None,
        description=(
            "An alternative to sampling with temperature, called nucleus sampling, where the model considers the "
            "results of the tokens with `top_p` probability mass. So `0.1` means only the tokens comprising the top "
            "10% probability mass are considered. We generally recommend altering this or `temperature` but not both."
        ),
    )
    n: int | None = Field(
        None,
        description=(
            "How many chat completion choices to generate for each input message. Keep `n` as `1` to receive one choice per input message."
        ),
    )
    stream: bool | None = Field(
        None,
        description=(
            "If set to `true`, the model response data will be streamed to the client as it is generated using server-sent events."
        ),
    )
    max_completion_tokens: int | None = Field(
        None,
        description=(
            "An upper bound for the number of tokens that can be generated for a completion, including visible "
            "output tokens and reasoning tokens."
        ),
    )
    max_tokens: int | None = Field(
        None,
        description=(
            "The maximum number of tokens that can be generated in the chat completion. This value can be "
            "used to control costs for text generated via API.\n\n This value is now deprecated in favor of "
            "`max_completion_tokens`, and is not compatible with o-series models."
        ),
        deprecated=True,
    )
    response_format: ResponseFormat | None = Field(
        None,
        description=(
            "An object specifying the format that the model must output.\n\n Setting to "
            '`{"type": "json_schema", "json_schema": {...}}` enables Structured Outputs which ensures the model will '
            'match your supplied JSON schema. \n\n Setting to `{"type": "json_object"}` enables the older JSON mode, '
            "which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for "
            "models that support it."
        ),
    )
    seed: int | None = Field(
        None,
        description=(
            "If specified, our system will make a best effort to sample deterministically, such that repeated "
            "requests with the same seed and parameters should return the same result. Determinism is not guaranteed, "
            "and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend."
        ),
    )
    stop: str | list[str] | None = Field(
        None,
        description=(
            "Might not be supported with latest reasoning models like `o3` and `o4-mini`.\n\n Up to 4 sequences where "
            "the API will stop generating further tokens. The returned text will not contain the stop sequence."
        ),
    )
    user: str | None = Field(
        None,
        description=("A stable identifier for your end-users. Used to boost cache hit rates by better bucketing similar requests."),
    )

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
    input: str | list[str] = Field(description="The input text(s) to embed")
    model: str = Field(description="ID of the model to use for embedding", examples=["llama3"])
    dimensions: int | None = Field(None, description="The number of dimensions to return for each embedding")
    encoding_format: str | None = Field(None, description="The encoding format of the embeddings", examples=["float"])
    user: str | None = Field(None, description="A unique identifier for the end-user", examples=["user123"])

    model_config = {"json_schema_extra": {"examples": [{"input": "Hello, how are you?", "model": "llama3", "encoding_format": "float"}]}}


class ImagesRequest(BaseModel):
    prompt: str = Field()
    background: Literal["auto", "transparent", "opaque"] | None = Field(default=None)
    model: str = Field()
    moderation: str | None = Field(default=None)
    n: int | None = Field(None, ge=1, le=4)
    output_compression: int | None = Field(95, ge=0, le=100)
    output_format: Literal["png", "webp", "jpeg"] | None = Field(default=None)
    quality: Literal["low", "medium", "high", "auto"] | None = Field(default=None)
    response_format: Literal["url", "b64_json"] | None = Field(default=None)
    size: str | None = Field(default=None)
    style: str | None = Field(default=None)
    user: str | None = Field(default=None)
    partial_images: int | None = Field(default=None)
    stream: bool | None = Field(default=None)


class CreateSpeechRequest(BaseModel):
    input: str = Field(..., max_length=4096, description="The text to generate audio for. The maximum length is 4096 characters.")
    model: str = Field(..., description="One of the available TTS models.")
    voice: str | None = Field(default=None, description="The voice to use when generating the audio.")
    instructions: str | None = Field(default=None, description="Control the voice of your generated audio with additional instructions.")
    format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None = Field(
        default=None,
        description="The format to audio in. Supported formats are mp3, opus, aac, flac, wav, and pcm.",
    )
    speed: float | None = Field(
        default=None, ge=0.25, le=4.0, description="The speed of the generated audio. Select a value from 0.25 to 4.0. 1.0 is the default."
    )
    stream_format: Literal["sse", "audio"] | None = Field(
        default=None,
        description="The format to stream the audio in. Supported formats are sse and audio. sse is not supported for tts-1 or tts-1-hd.",
    )


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
    type: Literal["server_vad"] = Field(..., description="Must be set to 'server_vad' to enable manual chunking using server side VAD")

    prefix_padding_ms: int | None = Field(
        default=None,
        ge=0,
        description="Amount of audio to include before the VAD detected speech (in milliseconds). "
        "This ensures the beginning of speech is not cut off.",
    )

    silence_duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Duration of silence to detect speech stop (in milliseconds). "
        "With shorter values the model will respond more quickly, "
        "but may jump in on short pauses from the user.",
    )

    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Sensitivity threshold (0.0 to 1.0) for voice activity detection. "
        "A higher threshold will require louder audio to activate the model, "
        "and thus might perform better in noisy environments.",
    )


class FormSerializable(BaseModel):
    @abstractmethod
    async def to_form(self, remove_model: bool, rewrite_model_to: str | None) -> FormData:
        """Serialize to FormData."""


class CreateTranscriptionRequest(FormSerializable):
    model: str = Field(..., description="ID of the model to use.")
    file: UploadFile = FastApiFile(
        ...,
        description="The audio file object (not file name) in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, "
        "while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability "
        "to automatically increase the temperature until certain thresholds are hit.",
    )
    type: str | None = Field(default=None, description="The format type, either 'text' or 'json_object'")
    chunking_strategy: Literal["auto"] | AudioChunkingStrategy | None = Field(
        default=None,
        description="Controls how the audio is cut into chunks."
        'When set to "auto", the server first normalizes loudness and'
        " then uses voice activity detection (VAD) to choose boundaries."
        " server_vad object can be provided to tweak VAD detection parameters manually."
        " If unset, the audio is transcribed as a single block.",
    )
    include: list[TranscriptionInclude] | None = Field(
        default=None,
        alias="include[]",
        description="Additional information to include in the transcription response. "
        "logprobs will return the log probabilities of the tokens in the response to "
        "understand the model's confidence in the transcription.",
    )
    language: str | None = Field(
        default=None,
        description="The language of the input audio. Supplying the input language in ISO-639-1 (e.g. en) "
        "format will improve accuracy and latency.",
    )
    prompt: str | None = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment. "
        "The prompt should match the audio language.",
    )
    stream: bool | None = Field(
        default=None,
        description="If set to true, the model response data will be streamed to the client as it is generated using server-sent events.",
    )
    timestamp_granularities: list[Literal["word", "segment"]] | None = Field(
        default=None,
        alias="timestamp_granularities[]",
        description="The timestamp granularities to populate for this transcription. response_format must be set verbose_json to use "
        "timestamp granularities. "
        "Either or both of these options are supported: word, or segment. Note: There is no additional latency for segment timestamps, "
        "but generating word timestamps incurs additional latency.",
    )

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

    include_obfuscation: bool | None = Field(
        default=None, description="Enable stream obfuscation to normalize payload sizes as mitigation for side-channel attacks."
    )
    include_usage: bool | None = Field(default=None, description="Include token usage statistics in the stream.")


class CompletionLegacyRequest(BaseModel):
    model: str = Field(description="ID of the model to use.")

    prompt: str | list[str] | list[int] | list[list[int]] = Field(
        default=...,
        description="The prompt(s) to generate completions for - can be string, array of strings, tokens, or token arrays.",
    )

    best_of: int | None = Field(
        default=None, ge=1, description="Generate best_of completions and return the best one. Must be greater than n if both are set."
    )

    echo: bool | None = Field(default=None, description="Echo back the prompt in addition to the completion.")

    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Penalize new tokens based on their existing frequency (-2.0 to 2.0)."
    )

    logit_bias: dict[str, Bias] | None = Field(
        default=None, description="Modify likelihood of specified tokens appearing (token ID -> bias from -100 to 100)."
    )

    logprobs: int | None = Field(default=None, ge=0, le=5, description="Include log probabilities on the most likely tokens (max 5).")

    max_tokens: int | None = Field(default=None, ge=1, description="Maximum number of tokens to generate in the completion.")

    n: int | None = Field(default=None, ge=1, description="How many completions to generate for each prompt.")

    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Penalize new tokens based on whether they appear in the text so far (-2.0 to 2.0)."
    )

    seed: int | None = Field(default=None, description="Seed for deterministic sampling (best effort, not guaranteed).")

    stop: str | StopSequenceList | None = Field(default=None, description="Up to 4 sequences where the API will stop generating.")

    stream: bool | None = Field(default=None, description="Stream back partial progress as server-sent events.")

    stream_options: StreamOptions | None = Field(default=None, description="Options for streaming response (only used when stream=True).")

    suffix: str | None = Field(default=None, description="Suffix after completion of inserted text.")

    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature (0-2). Higher = more random, lower = more deterministic."
    )

    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling - consider tokens with top_p probability mass (0-1)."
    )

    user: str | None = Field(default=None, description="Unique identifier for end-user to help monitor and detect abuse.")
