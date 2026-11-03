# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Models for chat completions."""

import re
from abc import abstractmethod
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from aiohttp import FormData
from fastapi import File as FastApiFile
from fastapi import UploadFile
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from server.utils.logger import logger

# `ChatCompletionEndpoint` callback field -> (registration type suffix, public endpoint path). The registry derives both an LLM's
# registration type and its advertised `props.endpoints` from the callbacks actually set, so a model never lists an endpoint it can't serve.
LLM_CALLBACK_MAP: tuple[tuple[str, str, str], ...] = (
    ("on_completion", "v1", "/v1/completions"),
    ("on_chat_completion", "v2", "/v1/chat/completions"),
    ("on_responses", "v3", "/v1/responses"),
    ("on_messages", "ant", "/v1/messages"),
    ("on_ollama_chat", "ollama", "/api/chat"),
)
ALL_LLM_SUFFIXES = tuple(suffix for _, suffix, _ in LLM_CALLBACK_MAP)
EMBEDDINGS_ENDPOINTS = ["/v1/embeddings"]
STT_ENDPOINTS = ["/v1/audio/transcriptions"]
TTS_ENDPOINTS = ["/v1/audio/speech"]
IMG_ENDPOINTS = ["/v1/images/generations"]
DECISION_ENDPOINTS = ["/v1/systemone"]

# Caps on the flat dict fields (`chat_template_kwargs`, `repetition_detection`). They are precautionary, not derived from any backend.
# Numeric bounds on the extra sampling fields follow the OpenAI/vLLM specs; `thinking_token_budget` and `top_k` have no upper bound
# on purpose, since the effective ceiling is the model's context window or vocabulary, which DF-Infra can't know here.
_FLAT_PARAM_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_FLAT_PARAM_MAX_STR_LENGTH = 1024

# Backend-specific fields accepted on chat completions. Any other field not declared on the request models is dropped. These are
# forwarded only to proxies with `ProxyOptions.forward_extra_fields` (self-hosted backends and mesh peers), because remote
# providers such as OpenAI or Gemini reject unknown parameters with 400.
CHAT_COMPLETION_EXTRA_FIELDS = frozenset(
    {
        "thinking_token_budget",
        "chat_template_kwargs",
        "repetition_detection",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "parallel_tool_calls",
    }
)
# `presence_penalty`, `frequency_penalty` and `logprobs` were already part of the legacy completions schema and stay forwarded
# to every backend, as before.
COMPLETION_LEGACY_EXTRA_FIELDS = frozenset({"thinking_token_budget", "repetition_detection", "top_k", "min_p", "repetition_penalty"})


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
    json_schema: dict[str, Any] | None = None


type FlatParams = dict[str, bool | int | float | str | None]


def _validate_flat_params(max_keys: int) -> AfterValidator:
    """Build a validator accepting only a small, flat dict of scalar values (no nesting, no long strings)."""

    def validate(value: FlatParams | None) -> FlatParams | None:
        if value is None:
            return value
        if len(value) > max_keys:
            msg = f"at most {max_keys} keys are allowed"
            raise ValueError(msg)
        for key, item in value.items():
            if not _FLAT_PARAM_KEY.match(key):
                msg = f"invalid key {key!r}"
                raise ValueError(msg)
            if isinstance(item, str) and len(item) > _FLAT_PARAM_MAX_STR_LENGTH:
                msg = f"value of {key!r} is longer than {_FLAT_PARAM_MAX_STR_LENGTH} characters"
                raise ValueError(msg)
        return value

    return AfterValidator(validate)


_THINKING_TOKEN_BUDGET_DESCRIPTION = (
    "Maximum number of thinking tokens before the model must answer (-1 means no limit). Self-hosted backends only. "
    "vLLM rejects values other than -1 unless the model is started with `--reasoning-parser` (set it via the model's extra args)."
)


def _reject_zero_top_k(value: int | None) -> int | None:
    if value == 0:
        msg = "top_k=0 is not accepted by every backend (SGLang rejects it); use -1 to disable top_k"
        raise ValueError(msg)
    return value


type TopK = Annotated[
    int | None,
    Field(ge=-1, description="Sample only from the `top_k` most likely tokens (-1 disables; 0 is rejected)."),
    AfterValidator(_reject_zero_top_k),
]


class ChatCompletionReasoningConfig(BaseModel):
    effort: Annotated[
        Literal["high", "medium", "low", "max", "none"],
        Field(description="Constrains effort on reasoning for reasoning models. Mirrors the top-level `reasoning_effort` field."),
    ] = "none"


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
            ge=0.0,
            le=2.0,
            description=(
                "What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more "
                "random, while lower values like `0.2` will make it more focused and deterministic. We generally recommend "
                "altering this or `top_p` but not both."
            ),
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "An alternative to sampling with temperature, called nucleus sampling, where the model considers the "
                "results of the tokens with `top_p` probability mass. So `0.1` means only the tokens comprising the top "
                "10% probability mass are considered. We generally recommend altering this or `temperature` but not both."
            ),
        ),
    ] = None
    n: Annotated[
        int | None,
        Field(
            ge=0,
            description=(
                "How many chat completion choices to generate for each input message."
                "Keep `n` as `1` to receive one choice per input message."
            ),
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
            ge=1,
            description=(
                "An upper bound for the number of tokens that can be generated for a completion, including visible "
                "output tokens and reasoning tokens."
            ),
        ),
    ] = None
    max_tokens: Annotated[
        int | None,
        Field(
            ge=1,
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
    reasoning_effort: Annotated[
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None,
        Field(
            description=(
                "Constrains effort on reasoning for reasoning models. Supported by backends that expose reasoning/thinking "
                "models (e.g. Ollama with deepseek-r1, qwq). Note that some reasoning models think by default even when "
                "this field is omitted - use `none` to explicitly disable reasoning rather than relying on its absence. "
                "AI SDK clients must set this via `providerOptions.<provider>.reasoning_effort`, since the SDK's portable "
                "top-level `reasoning` parameter is not automatically translated for non-recognized providers like Ollama."
            )
        ),
    ] = None
    reasoning: Annotated[
        ChatCompletionReasoningConfig | None,
        Field(
            description=(
                'Alternative, nested form of `reasoning_effort` (`{"effort": ...}`), accepted for parity with backends '
                "(e.g. Ollama) that support both forms. If both `reasoning_effort` and `reasoning.effort` are set, both are "
                "forwarded to the backend as-is; DeepFellow does not resolve conflicts between them."
            )
        ),
    ] = None

    thinking_token_budget: Annotated[
        int | None,
        Field(
            ge=-1,
            description=_THINKING_TOKEN_BUDGET_DESCRIPTION,
        ),
    ] = None
    chat_template_kwargs: Annotated[
        FlatParams | None,
        Field(description='Extra keyword arguments for the chat template, e.g. `{"enable_thinking": false}`. At most 16 scalar entries.'),
        _validate_flat_params(16),
    ] = None
    repetition_detection: Annotated[
        FlatParams | None,
        Field(description="Parameters of the backend's repetition detection. At most 8 scalar entries."),
        _validate_flat_params(8),
    ] = None
    top_k: TopK = None
    min_p: Annotated[float | None, Field(ge=0.0, le=1.0, description="Minimum token probability relative to the most likely token.")] = None
    repetition_penalty: Annotated[
        float | None, Field(gt=0.0, le=2.0, description="Penalty for repeating tokens, 1.0 means no penalty (0 exclusive to 2.0).")
    ] = None
    presence_penalty: Annotated[
        float | None,
        Field(ge=-2.0, le=2.0, description="Penalize new tokens based on whether they appear in the text so far (-2.0 to 2.0)."),
    ] = None
    frequency_penalty: Annotated[
        float | None, Field(ge=-2.0, le=2.0, description="Penalize new tokens based on their existing frequency (-2.0 to 2.0).")
    ] = None
    logprobs: Annotated[bool | None, Field(description="Return the log probability of each output token.")] = None
    parallel_tool_calls: Annotated[bool | None, Field(description="Whether to allow the model to run tool calls in parallel.")] = None

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
                },
                {
                    "model": "deepseek-r1",
                    "messages": [
                        {"role": "user", "content": "Explain quantum entanglement in one paragraph."},
                    ],
                    "reasoning_effort": "medium",
                    "stream": False,
                },
            ]
        }
    }


class EmbeddingRequest(BaseModel):
    input: Annotated[str | list[str] | list[int] | list[list[int]], Field(description="The input text(s) to embed")]
    model: Annotated[str, Field(description="ID of the model to use for embedding", examples=["llama3"])]
    dimensions: Annotated[int | None, Field(ge=1, description="The number of dimensions to return for each embedding")] = None
    encoding_format: Annotated[
        Literal["base64", "float"] | None, Field(description="The encoding format of the embeddings", examples=["float"])
    ] = None
    user: Annotated[str | None, Field(description="A unique identifier for the end-user", examples=["user123"])] = None

    model_config = {"json_schema_extra": {"examples": [{"input": "Hello, how are you?", "model": "llama3", "encoding_format": "float"}]}}


class ImagesRequest(BaseModel):
    prompt: str
    background: Literal["auto", "transparent", "opaque"] | None = None
    model: str
    moderation: str | None = None
    n: Annotated[int | None, Field(ge=1, le=10)] = None
    output_compression: Annotated[int | None, Field(ge=0, le=100)] = 95
    output_format: Literal["png", "webp", "jpeg"] | None = None
    quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None = None
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
    instructions: Annotated[
        str | None, Field(max_length=4096, description="Control the voice of your generated audio with additional instructions.")
    ] = None
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


class McpToolInfo(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = {}


class ModelProps(BaseModel):
    private: bool
    type: str
    # For LLMs the registry overwrites this with the endpoints actually registered for the model, so services leave it empty.
    endpoints: list[str] = []
    context_window: Annotated[int, Field(gt=0)] | None = None
    max_context_window: Annotated[int, Field(gt=0)] | None = None
    prefix: str | None = None
    transport: str | None = None
    tools: list[McpToolInfo] = []
    # `None` means the capability is unknown/not yet resolved for this backend; `True`/`False` means resolved.
    supports_vision: bool | None = None
    supports_audio: bool | None = None


type ModelType = Literal["llm", "embedding", "tts", "stt", "rerank", "txt2img", "decision", "mcp", "custom"]

type ModelId = str
type RegistrationId = str


_MAX_REASONABLE_CAPACITY = 100_000


class Model(BaseModel):
    id: RegistrationId
    name: ModelId
    type: ModelType
    props: ModelProps
    usage: int
    capacity: int | None = None
    capacity_known: bool = False
    warm: bool = False
    healthy: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_non_positive_capacity_to_unknown(cls, data: Any) -> Any:  # noqa: ANN401
        # A peer is untrusted input: a malformed/buggy `capacity` (e.g. 0) must degrade to
        # "unknown" (capacity=None, capacity_known=False) rather than crash mesh sync, unlike
        # our own internal RegisteredModel construction, which raises on the same condition to
        # catch the bug at its source. This must run on the *raw* input, before capacity=None
        # is normalized away, so a genuinely `capacity=None, capacity_known=True` peer (meaning
        # "unbounded", e.g. a cloud/proxy backend) is left untouched instead of being coerced
        # down to "unknown" alongside the malformed case.
        if not isinstance(data, dict):
            return data
        capacity = data.get("capacity")
        if capacity is None:
            return data
        # Peers may serialize capacity as a JSON float (5.0) or numeric string ("5", "5.0")
        # depending on what touched the payload upstream; those are coercible and must reach
        # pydantic's normal int coercion rather than being treated the same as a genuinely
        # malformed value. A bool is technically an int subclass but never a legitimate capacity,
        # so it's excluded up front rather than silently coerced (True/False -> 1/0). Only a
        # value that isn't even numeric, or is non-positive once coerced, counts as malformed here.
        try:
            coerced = None if isinstance(capacity, bool) else int(float(capacity))
        except (TypeError, ValueError, OverflowError):
            coerced = None
        # No real backend serves more than a few thousand concurrent requests; anything past this
        # is a corrupted/malicious value, not a legitimate capacity.
        if coerced is None or coerced <= 0 or coerced > _MAX_REASONABLE_CAPACITY or float(coerced) != float(capacity):
            logger.warning(
                "Peer model %r reported a non-positive or non-numeric capacity (%r); coercing to unknown",
                data.get("name") or data.get("id"),
                capacity,
            )
            data = {**data, "capacity": None, "capacity_known": False}
        elif not data.get("capacity_known", False):
            # A valid, in-range capacity implies capacity_known=True: a peer that reports a real
            # value but leaves capacity_known unset/false is inconsistent, not "unbounded" or
            # "unknown" — force it consistent rather than silently trusting the mismatched flag.
            data = {**data, "capacity_known": True}
        return data


class ApiModelProps(ModelProps):
    """`ModelProps` as reported by `/v1/models`, with fields derived from the whole registry entry."""

    alias_of: list[str] = []
    """Names of the models this entry is an alias of; renames are never reported."""


class ApiModel(BaseModel):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    props: ApiModelProps
    usage: int = 0
    """Requests in flight for this model, summed across its registrations.

    Live for local models and close to live (with some lag) for models from child nodes. For models received from a
    parent/ancestor (`owned_by="mesh-ancestor"`) it is only the value reported when the model was registered, because
    usage updates are sent upward only and are never refreshed downward.
    """
    capacity: int | None = None
    """Total concurrency limit summed across registrations; `None` if any registration is unbounded or unknown."""


class ApiModels(BaseModel):
    object: Literal["list"] = "list"
    data: list[ApiModel]


class ApiModelCompatible(BaseModel):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str


class ApiModelsCompatible(BaseModel):
    object: Literal["list"] = "list"
    data: list[ApiModelCompatible]


type TranscriptionInclude = Literal["logprobs"]

type Languages = Literal[
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


type ResponseTranscriptionFormat = Literal["text", "json", "verbose_json", "srt", "vtt"]


class CreateTranscriptionRequest(FormSerializable):
    file: UploadFile = FastApiFile(
        description="The audio file object (not file name) in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.",
    )
    model: Annotated[str, Field(description="ID of the model to use.")]
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
    known_speaker_names: Annotated[
        list[str] | None,
        Field(
            alias="known_speaker_names[]",
            description=(
                "Optional list of speaker names that correspond to the audio samples provided in "
                "known_speaker_references[]. Each entry should be a short identifier "
                "(for example customer or agent)."
            ),
        ),
    ] = None
    known_speaker_references: Annotated[
        list[str] | None,
        Field(
            alias="known_speaker_references[]",
            description=(
                "Optional list of audio samples (as data URLs) that contain known speaker references "
                "matching known_speaker_names[]. Each sample must be between 2 and 10 seconds, "
                "and can use any of the same input audio formats supported by file."
            ),
        ),
    ] = None
    language: Annotated[
        Languages | None,
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
    response_format: Annotated[
        ResponseTranscriptionFormat | None,
        Field(description=("The format of the output, in one of these options: json, text, srt, verbose_json, vtt, or diarized_json.")),
    ] = None
    stream: Annotated[
        bool | None,
        Field(
            description=(
                "If set to true, the model response data will be streamed to the client as it is generated using server-sent events."
            )
        ),
    ] = None
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
        int | None,
        Field(ge=0, le=20, description="Generate best_of completions and return the best one. Must be greater than n if both are set."),
    ] = None
    echo: Annotated[bool | None, Field(description="Echo back the prompt in addition to the completion.")] = None
    frequency_penalty: Annotated[
        float | None, Field(ge=-2.0, le=2.0, description="Penalize new tokens based on their existing frequency (-2.0 to 2.0).")
    ] = None
    logit_bias: Annotated[
        dict[str, Bias] | None, Field(description="Modify likelihood of specified tokens appearing (token ID -> bias from -100 to 100).")
    ] = None
    logprobs: Annotated[int | None, Field(ge=0, le=5, description="Include log probabilities on the most likely tokens (max 5).")] = None
    max_tokens: Annotated[int | None, Field(ge=0, description="Maximum number of tokens to generate in the completion.")] = None
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
    thinking_token_budget: Annotated[
        int | None,
        Field(
            ge=-1,
            description=_THINKING_TOKEN_BUDGET_DESCRIPTION,
        ),
    ] = None
    repetition_detection: Annotated[
        FlatParams | None,
        Field(description="Parameters of the backend's repetition detection. At most 8 scalar entries."),
        _validate_flat_params(8),
    ] = None
    top_k: TopK = None
    min_p: Annotated[float | None, Field(ge=0.0, le=1.0, description="Minimum token probability relative to the most likely token.")] = None
    repetition_penalty: Annotated[
        float | None, Field(gt=0.0, le=2.0, description="Penalty for repeating tokens, 1.0 means no penalty (0 exclusive to 2.0).")
    ] = None


type SortOrder = Literal["asc", "desc"]


class McpAllowedToolsFilter(BaseModel):
    tool_names: list[str]


class InputImageMask(BaseModel):
    file_id: Annotated[str | None, Field(description="File ID for mask image")]
    image_url: Annotated[str | None, Field(description="Base64 encoded image mask")]


class FunctionDef(BaseModel):
    type: Literal["function"] = "function"
    name: str
    parameters: dict[str, Any]
    strict: bool = False
    description: str = ""


class McpToolDef(BaseModel):
    type: Literal["mcp"] = "mcp"
    server_label: Annotated[str, Field(description="Label for MCP server identification", examples=["git mcp"])]
    server_url: Annotated[str, Field(description="URL for MCP server endpoint", examples=["http://127.0.0.1:8001/mcp"])]
    allowed_tools: Annotated[list[str] | McpAllowedToolsFilter | None, Field(description="List of allowed tool names or filter object")] = (
        None
    )
    headers: Annotated[dict[str, str] | None, Field(description="HTTP headers to send")] = None
    require_approval: Annotated[
        dict[str, str] | str | None, Field(description="Tools requiring approval. Defaults to 'always'", examples=["always", "never"])
    ] = "always"
    server_description: str = ""


class ImageGenToolDef(BaseModel):
    type: Literal["image_generation"] = "image_generation"
    background: Annotated[Literal["auto", "transparent", "opaque"] | None, Field(description="Background type for generated image")] = (
        "auto"
    )
    input_image_mask: Annotated[
        InputImageMask | None, Field(description="Mask for inpainting operation containing image URL or file ID")
    ] = None
    model: Annotated[str, Field(description="Generation model to use")]
    moderation: Annotated[Literal["auto"] | None, Field(description="Moderation level")] = "auto"
    output_compression: Annotated[int | None, Field(ge=0, le=100, description="Output compression level (0-100)")] = 100
    output_format: Annotated[Literal["png", "webp", "jpeg"] | None, Field(description="Output format for generated image")] = "png"
    partial_images: Annotated[int | None, Field(ge=0, le=3, description="Number of partial images to generate in streaming mode (0-3)")] = 0
    quality: Annotated[Literal["low", "medium", "high", "auto"] | None, Field(description="Quality level for image")] = "auto"
    size: Annotated[str | None, Field(description="Generated image dimensions (pixel)", examples=["1024x1024", "auto"])] = "auto"


class ComparisonFilter(BaseModel):
    """A filter used to compare a specified attribute key to a given value using a defined comparison operation."""

    key: Annotated[str, Field(description="The key to compare against the value")]
    type: Annotated[
        Literal["eq", "ne", "gt", "gte", "lt", "lte"],
        Field(
            description="Specifies the comparison operator: eq (equals), ne (not equal), gt (greater than), "
            "gte (greater than or equal), lt (less than), lte (less than or equal)"
        ),
    ]
    value: Annotated[
        str | int | float | bool,
        Field(description="The value to compare against the attribute key; supports string, number, or boolean types"),
    ]

    def matches(self, attributes: dict[str, Any]) -> bool:
        """Check if this filter matches the given attributes."""
        attr_value = attributes.get(self.key)

        if attr_value is None:
            return False

        if self.type == "eq":
            return attr_value == self.value
        if self.type == "ne":
            return attr_value != self.value
        if self.type == "lt":
            return attr_value < self.value
        if self.type == "lte":
            return attr_value <= self.value
        if self.type == "gt":
            return attr_value > self.value
        if self.type == "gte":
            return attr_value >= self.value

        return False  # pragma: no cover


class CompoundFilter(BaseModel):
    """Combine multiple filters using 'and' or 'or'."""

    filters: Annotated[
        list[Union["ComparisonFilter", "CompoundFilter"]],
        Field(description="Array of filters to combine. Items can be ComparisonFilter or CompoundFilter"),
    ]
    type: Annotated[Literal["and", "or"], Field(description="Type of operation: 'and' or 'or'")]

    def matches(self, attributes: dict[str, Any]) -> bool:
        """Check if this compound filter matches the given attributes."""
        if self.type == "and":
            return all(f.matches(attributes) for f in self.filters)

        return any(f.matches(attributes) for f in self.filters)


CompoundFilter.model_rebuild()


class RankingOptions(BaseModel):
    """Ranking options for search."""

    ranker: Annotated[str | None, Field(description="Ranking algorithm to use")] = "auto"
    score_threshold: Annotated[float | None, Field(description="Minimum score threshold for results")] = 0.0


class FileSearchToolDef(BaseModel):
    type: Literal["file_search"] = "file_search"
    vector_store_ids: Annotated[list[str], Field(description="List of vector store IDs to search")]
    filters: Annotated[CompoundFilter | ComparisonFilter | None, Field(description="Filter to apply to results")] = None
    max_num_results: Annotated[int, Field(ge=1, le=50, description="Maximum number of results")] = 25
    ranking_options: Annotated[RankingOptions | None, Field(description="Ranking options")] = None


ToolResponsesInDF = FileSearchToolDef | McpToolDef | ImageGenToolDef
ToolResponses = ToolResponsesInDF | FunctionDef


class HostedToolChoice(BaseModel):
    type: Literal["file_search", "web_search_preview", "computer_use_preview", "code_interpreter", "image_generation"]


class McpToolChoice(BaseModel):
    type: Literal["mcp"] = "mcp"
    server_label: str
    name: str = ""


class UserLocation(BaseModel):
    type: Literal["approximate"]
    city: str = ""
    country: str = ""
    region: str = ""
    timezone: str = ""


# Request


class Message(BaseModel):
    type: Literal["message"] = "message"
    content: str
    role: Literal["user", "assistant", "system", "developer"]


class InputText(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class InputImage(BaseModel):
    type: Literal["input_image"] = "input_image"
    file_id: str = ""
    image_url: str = ""
    detail: str = "auto"


class InputFile(BaseModel):
    type: Literal["input_file"] = "input_file"
    file_data: str = ""
    file_id: str = ""
    file_url: str = ""
    filename: str = ""


InputMessageContent = InputText | InputImage | InputFile


class InputMessage(BaseModel):
    type: Literal["message"] = "message"
    content: list[InputMessageContent]
    role: Literal["user", "assistant", "system", "developer"]
    status: Literal["in_progress", "completed", "incomplete"] = "completed"


class TopLogprobsDetail(BaseModel):
    bytes: list[int]
    logprob: float
    token: str


class LogprobsDetail(BaseModel):
    bytes: list[int]
    logprob: float
    token: str
    top_logprobs: list[TopLogprobsDetail]


class FileCitation(BaseModel):
    type: Literal["file_citation"] = "file_citation"
    file_id: str
    filename: str
    index: int


class UrlCitation(BaseModel):
    type: Literal["url_citation"] = "url_citation"
    title: str
    url: str
    end_index: int
    start_index: int


class ContainerFileCitation(BaseModel):
    type: Literal["container_file_citation"] = "container_file_citation"
    container_id: str
    file_id: str
    filename: str
    start_index: int
    end_index: int


class FilePath(BaseModel):
    type: Literal["file_path"] = "file_path"
    file_id: str
    index: int


class OutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: list[FileCitation | UrlCitation | ContainerFileCitation | FilePath] = []
    logprobs: list[LogprobsDetail] = []


class Refusal(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str


class OutputMessage(BaseModel):
    id: str | None = None
    type: Literal["message"] = "message"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    content: list[OutputText | Refusal] = []
    role: Literal["assistant"] = "assistant"


class ItemReference(BaseModel):
    id: str
    type: Literal["item_reference"] = "item_reference"


class FileSearchResult(BaseModel):
    attributes: dict[str, str | bool | int | float] = {}
    file_id: str = ""
    filename: str = ""
    score: float = 0
    text: str = ""


class FileSearchToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["file_search_call"] = "file_search_call"
    status: Literal["in_progress", "searching", "incomplete", "failed"] = "searching"
    queries: list[str]
    results: list[FileSearchResult] = []


class FunctionToolCall(BaseModel):
    id: str | None = None
    type: Literal["function_call"] = "function_call"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    call_id: str | None = None
    name: str
    arguments: str


class ReasoningSummary(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ReasoningContentItem(BaseModel):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class Reasoning(BaseModel):
    id: str | None = None
    type: Literal["reasoning"] = "reasoning"
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    summary: list[ReasoningSummary]
    content: list[ReasoningContentItem] | None = None
    encrypted_content: str | None = None


class ImageGenerationCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["image_generation_call"] = "image_generation_call"
    result: str = ""
    status: str = "completed"


class ToolResponse(BaseModel):
    name: str
    descriptions: str = ""
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = {}


class McpListTools(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    server_label: str
    tools: list[ToolResponse] = []
    error: str = ""


class McpApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["mcp_approval_request"] = "mcp_approval_request"
    server_label: str
    name: str
    arguments: str


class McpApprovalResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["mcp_approval_response"] = "mcp_approval_response"
    approval_request_id: str
    approve: bool
    reason: str = ""


class McpToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["mcp_call"] = "mcp_call"
    server_label: str
    name: str
    arguments: str = ""
    output: str = ""
    error: str = ""


class Prompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    variables: dict[str, Any] = {}
    version: str = ""


class ReasoningConfig(BaseModel):
    effort: Annotated[
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None,
        Field(
            description=(
                "Constrains effort on reasoning for reasoning models. Currently supported values are none, minimal, low, "
                "medium, high, xhigh, and max."
            )
        ),
    ] = None
    summary: Annotated[
        Literal["auto", "concise", "detailed"] | None,
        Field(description="A summary of the reasoning performed by the model."),
    ] = None
    generate_summary: Annotated[
        Literal["auto", "concise", "detailed"] | None,
        Field(deprecated=True, description="Deprecated in favor of `summary`."),
    ] = None
    context: Annotated[
        Literal["auto", "current_turn", "all_turns"] | None,
        Field(description="Controls which reasoning items are rendered back to the model on later turns."),
    ] = None
    mode: Annotated[
        Literal["standard", "pro"] | None,
        Field(description="Controls the reasoning execution mode for the request."),
    ] = None


class TextFormat(BaseModel):
    type: Literal["text"] = "text"


class JsonSchemaFormat(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    name: str
    description: str = ""
    json_schema: Annotated[dict[str, Any], Field(alias="schema")]
    strict: bool = False


class JsonObjectFormat(BaseModel):
    type: Literal["json_object"] = "json_object"


class TextConfig(BaseModel):
    format: TextFormat | JsonSchemaFormat | JsonObjectFormat


class CustomToolCall(BaseModel):
    id: str | None = None
    type: Literal["custom_tool_call"] = "custom_tool_call"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    call_id: str | None = None
    namespace: str | None
    name: str
    input: str


class FunctionCallOutputTextInput(BaseModel):
    type: Literal["input_text"]
    text: str


class FunctionCallOutputImageInput(BaseModel):
    type: Literal["input_image"]
    detail: Literal["low", "high", "auto", "original"] = "auto"
    file_id: str | None = None
    image_url: str | None = None


class FunctionCallOutputFileInput(BaseModel):
    type: Literal["input_file"]
    file_data: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None


class FunctionCallOutput(BaseModel):
    id: str | None = None
    type: Literal["function_call_output"] = "function_call_output"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    call_id: str
    output: str | list[FunctionCallOutputTextInput | FunctionCallOutputImageInput | FunctionCallOutputFileInput]


class CustomToolCallOutput(BaseModel):
    id: str | None = None
    type: Literal["custom_tool_call_output"] = "custom_tool_call_output"
    call_id: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    created_by: str | None = None
    output: str | list[FunctionCallOutputTextInput | FunctionCallOutputImageInput | FunctionCallOutputFileInput]


class SearchSource(BaseModel):
    type: Literal["url"] = "url"
    url: str


class Search(BaseModel):
    type: Literal["search"] = "search"
    queries: list[str] = []
    sources: list[SearchSource] = []


class OpenPage(BaseModel):
    type: Literal["open_page"] = "open_page"
    url: str | None = None


class FindInPage(BaseModel):
    type: Literal["find_in_page"] = "find_in_page"
    pattern: str
    url: str


class WebSearchCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["web_search_call"] = "web_search_call"
    status: Literal["in_progress", "completed", "searching", "failed"] = "completed"
    action: Search | OpenPage | FindInPage


class Compaction(BaseModel):
    id: str
    type: Literal["compaction"] = "compaction"
    encrypted_content: str
    created_by: str | None = None


class ToolSearchCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    arguments: dict[str, str]
    call_id: str | None = Field(default_factory=lambda: str(uuid4()))
    execution: Literal["server", "client"]
    status: Literal["in_progress", "completed", "incomplete"]
    type: Literal["tool_search_call"]
    created_by: str | None = None


class ToolSearchOutput(BaseModel):
    type: Literal["tool_search_output"] = "tool_search_output"
    id: str = Field(default_factory=lambda: str(uuid4()))
    call_id: str | None = Field(default_factory=lambda: str(uuid4()))
    execution: Literal["server", "client"]
    status: Literal["in_progress", "completed", "incomplete"]
    tools: list[ToolResponses]
    created_by: str | None = None


class OperationCreateFile(BaseModel):
    type: Literal["create_file"] = "create_file"
    path: str
    diff: str


class OperationDeleteFile(BaseModel):
    type: Literal["delete_file"] = "delete_file"
    path: str


class OperationUpdateFile(BaseModel):
    type: Literal["update_file"] = "update_file"
    path: str
    diff: str


class ApplyPatchToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    status: Literal["in_progress", "completed"] = "completed"
    type: Literal["apply_patch_call"] = "apply_patch_call"
    created_by: str | None = None
    operation: OperationCreateFile | OperationDeleteFile | OperationUpdateFile


class ApplyPatchToolCallOutput(BaseModel):
    type: Literal["apply_patch_call_output"] = "apply_patch_call_output"
    id: str = Field(default_factory=lambda: str(uuid4()))
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    status: Literal["failed", "completed"] = "completed"
    created_by: str | None = None
    output: str | None = None


Input = (
    Message
    | InputMessage
    | OutputMessage
    | FileSearchToolCall
    | FunctionToolCall
    | CustomToolCall
    | Reasoning
    | ImageGenerationCall
    | McpListTools
    | McpApprovalRequest
    | McpApprovalResponse
    | ItemReference
    | McpToolCall
    | FunctionCallOutput
    | CustomToolCallOutput
    | WebSearchCall
    | Compaction
    | ToolSearchCall
    | ToolSearchOutput
    | ApplyPatchToolCall
    | ApplyPatchToolCallOutput
)


class ResponsesRequest(BaseModel):
    model: Annotated[
        str,
        Field(
            title="model id",
            description=(
                "Model ID used to generate the response, like `llama3.1:8b` or `qwen3:1.7b`."
                "DeepFellow supports a wide range of models with different capabilities, and performance characteristics."
            ),
        ),
    ]
    input: Annotated[
        str | list[Input] | None, Field(description="Text, image, or file inputs to the model, used to generate a response.")
    ] = None
    instructions: Annotated[
        str | None,
        Field(
            description=(
                "Previous messages inserted into the model's context."
                "When using along with `previous_response_id`, the instructions from a previous response will not be carried "
                "over to the next response. This makes it simple to swap out system (or developer) messages in new responses."
            ),
        ),
    ] = None
    text: Annotated[
        TextConfig | None,
        Field(description="Configuration options for a text response from the model. Can be plain text, JSON or structured JSON."),
    ] = None
    prompt: Annotated[Prompt | None, Field(description="Reference to a prompt template and its variables.")] = None
    tool_choice: Annotated[
        Literal["none", "auto", "required"] | HostedToolChoice | McpToolChoice | None,
        Field(
            description=(
                "[Currently not supported] How the model should select which tool (or tools) to use when generating a response. "
                "`none` is the default when no tools are present. `auto` is the default if tools are present. "
                "See the `tools` parameter below to see how to specify which tools the model can call."
            ),
        ),
    ] = None
    tools: Annotated[
        list[ToolResponses] | None,
        Field(
            description=(
                "An array of tools the model may call while generating a response. You can specify which tool to use by setting "
                "the `tool_choice` parameter."
                "We support the following categories of tools:"
                "- Built-in tools: Tools that are provided by DeepFellow that extend the model's capabilities, like web search "
                "or file search."
                "- MCP Tools: Integrations with third-party systems via custom MCP servers or "
                "predefined connectors such as Google Drive and SharePoint."
                "- Function calls (custom tools): Functions that are defined by you, "
                "enabling the model to call your own code with strongly "
                "typed arguments and outputs. You can also use custom tools to call your own code."
            ),
        ),
    ] = None
    temperature: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=2.0,
            description=(
                "What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more random, "
                "while lower values like `0.2` will make it more focused and deterministic. We generally recommend altering this "
                "or `top_p` but not both. Not supported by all models - reasoning-only models may reject this parameter entirely "
                "if it is present, so it is only forwarded when explicitly set."
            ),
        ),
    ] = None
    top_p: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            description=(
                "An alternative to sampling with `temperature`, called nucleus sampling, where the model considers the results of the "
                "tokens with `top_p` probability mass. So `0.1` means only the tokens comprising the top 10% probability mass "
                "are considered."
                "We generally recommend altering this or `temperature` but not both. Not supported by all models - reasoning-only "
                "models may reject this parameter entirely if it is present, so it is only forwarded when explicitly set."
            ),
        ),
    ] = None
    reasoning: Annotated[ReasoningConfig | None, Field(description="Configuration options for reasoning models.")] = None
    stream: Annotated[
        bool | None,
        Field(
            description=(
                "If set to true, the model response data will be streamed to the client as it is generated using server-sent events."
            ),
        ),
    ] = None
    max_output_tokens: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "An upper bound for the number of tokens that can be generated for a response, "
                "including visible output tokens and reasoning tokens."
            ),
        ),
    ] = None
    max_tool_calls: Annotated[
        int | None,
        Field(
            ge=0,
            le=10,
            description=(
                "The maximum number of total calls to built-in tools that can be processed in a response. "
                "This maximum number applies across all built-in tool calls, not per individual tool. "
                "Any further attempts to call a tool by the model will be ignored."
            ),
        ),
    ] = None
    parallel_tool_calls: Annotated[bool | None, Field(description="Whether to allow the model to run tool calls in parallel.")] = None
    include: Annotated[list[str] | None, Field(description="Specify additional output data to include in the model response.")] = None
    background: Annotated[bool | None, Field(description="Whether to run the model response in the background.")] = None
    previous_response_id: Annotated[
        str | None,
        Field(
            description=(
                "[Currently not supported] The unique ID of the previous response to the model. Use this to create multi-turn "
                "conversations. Cannot be used in conjunction with conversation."
            ),
        ),
    ] = None
    top_logprobs: Annotated[
        int | None,
        Field(
            ge=0,
            le=20,
            description=(
                "An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, "
                "each with an associated log probability."
            ),
        ),
    ] = None
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(
            description=(
                "Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information "
                "about the object in a structured format, and querying for objects via API or the dashboard."
            ),
        ),
    ] = None
    service_tier: Annotated[
        Literal["auto", "default", "flex", "priority"] | None,
        Field(deprecated=True, description="[Not supported. Placeholder for compatibility]"),
    ] = None
    store: Annotated[
        bool | None,
        Field(description="[Currently not supported] Whether to store the generated model response for later retrieval via API."),
    ] = None
    truncation: Annotated[
        Literal["auto", "disabled"] | None,
        Field(
            description=(
                "[Currently not supported] The truncation strategy to use for the model response."
                "- auto: If the input to this Response exceeds the model's context window size, the model will truncate the response to "
                "fit the context window by dropping items from the beginning of the conversation."
                "- disabled (default): If the input size will exceed the context window size for a model, the request will "
                "fail with a 400 error."
            ),
        ),
    ] = None
    user: Annotated[
        str | None,
        Field(
            deprecated=True,
            description=(
                "This field is being replaced by `safety_identifier` and `prompt_cache_key`. "
                "Use `prompt_cache_key` instead to maintain caching optimizations. A stable identifier for your end-users."
            ),
        ),
    ] = None

    model_config = {"json_schema_extra": {"examples": [{"model": "llama3.1:8b", "input": "Say Hello World o/."}]}}


class OllamaToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any]


class OllamaToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: OllamaToolFunction


class OllamaChatToolCallFunction(BaseModel):
    name: str
    description: str | None = None
    arguments: dict[str, Any] | None = None


class OllamaChatToolCall(BaseModel):
    function: OllamaChatToolCallFunction


class OllamaChatMessage(BaseModel):
    role: str  # Ollama native API accepts arbitrary roles (e.g. "document"), not just the well-known ones
    content: str  # Ollama native API supports list content for multimodal; str-only is a known limitation here
    images: list[str] | None = None
    tool_calls: list[OllamaChatToolCall] | None = None


class OllamaOptions(BaseModel):
    # Per-request options of Ollama's `api.Options`; `num_ctx` is kept but bounded by `max_context_tokens`.
    # Anything else - runner/hardware options (`num_gpu`, `num_thread`, `num_batch`, `main_gpu`, `use_mmap`) and options
    # Ollama no longer supports (`mirostat*`, `tfs_z`, `penalize_newline`) - is dropped, not forwarded, mirroring Ollama,
    # which ignores unknown options with a warning.
    model_config = ConfigDict(extra="ignore")

    num_ctx: Annotated[int | None, Field(ge=1)] = None
    # Ollama treats a negative or missing `num_predict` as "up to 10 x num_ctx"; Infra replaces it with
    # `max_generated_tokens` before forwarding (see `server/utils/request_limits.py`).
    num_predict: Annotated[int | None, Field(ge=-2)] = None
    num_keep: Annotated[int | None, Field(ge=-1)] = None
    repeat_last_n: Annotated[int | None, Field(ge=-1)] = None
    repeat_penalty: Annotated[float | None, Field(ge=0.0, le=10.0)] = None
    presence_penalty: Annotated[float | None, Field(ge=-2.0, le=2.0)] = None
    frequency_penalty: Annotated[float | None, Field(ge=-2.0, le=2.0)] = None
    temperature: Annotated[float | None, Field(ge=0.0, le=2.0)] = None
    seed: int | None = None
    stop: str | list[str] | None = None
    top_k: Annotated[int | None, Field(ge=0)] = None
    top_p: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    min_p: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    typical_p: Annotated[float | None, Field(ge=0.0, le=1.0)] = None


class OllamaChatRequest(BaseModel):
    model: Annotated[
        str,
        Field(
            description=(
                "Model ID used to generate the response, like gemma3 or llama3.2. "
                "Deepfellow supports a wide range of models with different capabilities, and performance "
                "characteristics."
            )
        ),
    ]
    messages: Annotated[
        list[OllamaChatMessage],
        Field(
            description=(
                "A list of messages comprising the conversation so far. Depending on the model you use, "
                "different message types (modalities) are supported, like text and images."
            )
        ),
    ]
    tools: Annotated[
        list[OllamaToolDefinition] | None,
        Field(
            description=(
                "A list of tools the model may call. Currently, only functions are supported as a tool. Use this to "
                "provide a list of functions the model may generate JSON inputs for."
            )
        ),
    ] = None
    think: Annotated[
        bool | Literal["high", "medium", "low"] | None,
        Field(description="For thinking models: whether the model should think before responding. Can be a boolean or effort level."),
    ] = None
    format: Annotated[
        Literal["json"] | dict[str, Any] | None,
        Field(description="The format to return a response in. Can be `json` or a JSON schema object."),
    ] = None
    options: Annotated[
        OllamaOptions | None,
        Field(description="Additional model parameters such as temperature, top_k, top_p and others."),
    ] = None
    stream: Annotated[
        bool,
        Field(description="If false the response will be returned as a single response object, rather than a stream of objects."),
    ] = True
    keep_alive: Annotated[
        str | int | None,
        Field(description="Controls how long the model will stay loaded into memory following the request (default: 5m)."),
    ] = None
    logprobs: Annotated[
        bool | None,
        Field(description="Return log probabilities of the output tokens."),
    ] = None
    top_logprobs: Annotated[
        int | None,
        Field(description="Number of most likely tokens to return at each token position."),
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
                    "stream": False,
                }
            ]
        }
    }


class ErrorDetail(BaseModel):
    """An error object returned when the model fails."""

    code: str
    message: str


class IncompleteDetails(BaseModel):
    """Details about why the response is incomplete."""

    reason: str


class InputTokensDetails(BaseModel):
    """Detailed breakdown of token usage."""

    cached_tokens: int | None = None


class OutputTokensDetails(BaseModel):
    """Detailed breakdown of token usage."""

    reasoning_tokens: int | None = None


class Usage(BaseModel):
    """Represents token usage details for the response."""

    input_tokens: int
    input_tokens_details: InputTokensDetails | None = None
    output_tokens: int
    output_tokens_details: OutputTokensDetails | None = None
    total_tokens: int


class ReasoningOutput(BaseModel):
    effort: Literal["low", "medium", "high"] | None = None
    summary: str = ""


Instruction = (
    Message
    | InputMessage
    | OutputMessage
    | FileSearchToolCall
    | FunctionToolCall
    | Reasoning
    | ImageGenerationCall
    | McpListTools
    | McpApprovalResponse
    | ItemReference
    | McpToolCall
    | CustomToolCall
    | FunctionCallOutput
    | CustomToolCallOutput
    | McpApprovalRequest
    | WebSearchCall
    | Compaction
    | ToolSearchCall
    | ToolSearchOutput
    | ApplyPatchToolCall
    | ApplyPatchToolCallOutput
)


Output = (
    OutputMessage
    | FunctionToolCall
    | FileSearchToolCall
    | Reasoning
    | ImageGenerationCall
    | McpListTools
    | McpApprovalRequest
    | McpToolCall
    | ItemReference
    | CustomToolCall
    | FunctionCallOutput
    | CustomToolCallOutput
    | McpApprovalRequest
    | WebSearchCall
    | Compaction
    | ToolSearchCall
    | ToolSearchOutput
    | ApplyPatchToolCall
    | ApplyPatchToolCallOutput
)


class ResponsesResponse(BaseModel):
    object: Literal["response"] = "response"
    output: list[Output] = []
    output_text: str = ""
    text: TextFormat | JsonSchemaFormat | JsonObjectFormat | None = None
    reasoning: ReasoningOutput | None = None
    tool_choice: Literal["none", "auto", "required"] | HostedToolChoice | McpToolChoice = "auto"
    model: str
    prompt: Prompt | None = None
    instructions: str | list[Instruction] = ""
    tools: list[ToolResponses] = []
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    parallel_tool_calls: bool = False
    background: bool = False
    temperature: float | None = None
    top_p: float | None = None
    truncation: Literal["auto", "disabled"] = "disabled"
    status: Literal["in_progress", "completed", "incomplete", "failed", "cancelled", "queued"] = "completed"
    created_at: int = int(datetime.now(tz=UTC).timestamp())
    top_logprobs: int | None = None
    metadata: dict[str, Any] = {}
    previous_response_id: str | None = None
    usage: Usage | None = None
    service_tier: str = ""
    user: str = ""
    id: str = ""
    error: ErrorDetail | None = None
    incomplete_details: IncompleteDetails | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
                    "object": "response",
                    "created_at": 1741476542,
                    "status": "completed",
                    "error": None,
                    "incomplete_details": None,
                    "instructions": None,
                    "max_output_tokens": None,
                    "model": "qwen3:1.7b",
                    "output": [
                        {
                            "type": "message",
                            "id": "67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": (
                                        "In a peaceful grove beneath a silver moon, "
                                        "a unicorn named Lumina discovered a hidden pool that reflected the stars. [...]"
                                    ),
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "parallel_tool_calls": True,
                    "previous_response_id": None,
                    "reasoning": {"effort": None, "summary": None},
                    "store": True,
                    "temperature": 1.0,
                    "text": {"format": {"type": "text"}},
                    "tool_choice": "auto",
                    "tools": [],
                    "top_p": 1.0,
                    "truncation": "disabled",
                    "usage": {
                        "input_tokens": 36,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 87,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 123,
                    },
                    "user": None,
                    "metadata": {},
                }
            ]
        }
    }


type MessagesRole = Literal["user", "assistant"]


class CacheControlEphemeral(BaseModel):
    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"]


class CitationCharLocationParam(BaseModel):
    type: Literal["char_location"]
    cited_text: str
    document_index: int
    document_title: str
    end_char_index: int
    start_char_index: int


class CitationPageLocationParam(BaseModel):
    type: Literal["page_location"]
    cited_text: str
    document_index: int
    document_title: str
    end_char_index: int
    start_char_index: int


class CitationContentBlockLocationParam(BaseModel):
    type: Literal["content_block_location"]
    cited_text: str
    document_index: int
    document_title: str
    end_char_index: int
    start_char_index: int


class CitationWebSearchResultLocationParam(BaseModel):
    type: Literal["web_search_result_location"]
    cited_text: str
    encrypted_index: str
    title: str
    url: str


class CitationSearchResultLocationParam(BaseModel):
    type: Literal["search_result_location"]
    cited_text: str
    end_block_index: int
    search_result_index: int
    source: str
    start_block_index: int
    title: str


type TextCitationParam = (
    CitationCharLocationParam
    | CitationPageLocationParam
    | CitationContentBlockLocationParam
    | CitationWebSearchResultLocationParam
    | CitationSearchResultLocationParam
)


class TextBlockParam(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: CacheControlEphemeral | None = None
    citations: list[TextCitationParam]


type MediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class Base64ImageSource(BaseModel):
    type: Literal["base64"]
    data: str
    media_type: MediaType


class URLImageSource(BaseModel):
    type: Literal["url"]
    url: str


class ImageBlockParam(BaseModel):
    type: Literal["image"]
    source: Base64ImageSource | URLImageSource
    cache_control: CacheControlEphemeral | None = None


class CitationsConfigParam(BaseModel):
    enabled: bool | None = None


class Base64PDFSource(BaseModel):
    type: Literal["base64"]
    media_type: Literal["application/pdf"]
    data: str


class PlainTextSource(BaseModel):
    type: Literal["text"]
    media_type: Literal["text/plain"]
    data: str


class ContentBlockSource(BaseModel):
    type: Literal["content"]
    content: str | list[TextBlockParam | ImageBlockParam]


class URLPDFSource(BaseModel):
    type: Literal["url"]
    url: str


class DocumentBlockParam(BaseModel):
    type: Literal["document"]
    source: Base64PDFSource | PlainTextSource | ContentBlockSource | URLPDFSource
    cache_control: CacheControlEphemeral | None = None
    citations: CitationsConfigParam | None = None
    context: str | None = None
    title: str | None = None


class SearchResultParam(BaseModel):
    type: Literal["search_result"]
    content: list[TextBlockParam]
    cache_control: CacheControlEphemeral | None = None
    citations: CitationsConfigParam | None = None


class ThinkingBlockParam(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str


class RedactedThinkingBlockParam(BaseModel):
    type: Literal["redacted_thinking"]
    data: str


class ToolUseBlockParam(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[Any, Any]
    cache_control: CacheControlEphemeral | None = None


class ToolResultBlockParam(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[TextBlockParam | ImageBlockParam]
    is_error: bool | None = None
    cache_control: CacheControlEphemeral | None = None


class ServerToolUseBlockParam(BaseModel):
    type: Literal["server_tool_use"]
    name: Literal["web_search"]
    id: str
    input: dict[Any, Any]
    cache_control: CacheControlEphemeral | None = None


class WebSearchToolResultBlockItem(BaseModel):
    type: Literal["web_search_result"]
    encrypted_content: str
    title: str
    url: str
    page_age: str | None = None


class WebSearchToolRequestError(BaseModel):
    type: Literal["web_search_tool_result_error"]
    error_code: Literal["invalid_tool_input", "unavailable", "max_uses_exceeded", "too_many_requests", "query_too_long"]


type WebSearchToolResultBlockContent = list[WebSearchToolResultBlockItem] | WebSearchToolRequestError


class WebSearchToolResultBlockParam(BaseModel):
    type: Literal["web_search_tool_result"]
    content: WebSearchToolResultBlockContent
    tool_use_id: str
    cache_control: CacheControlEphemeral | None = None


type ContentBlockParam = (
    TextBlockParam
    | ImageBlockParam
    | DocumentBlockParam
    | SearchResultParam
    | ThinkingBlockParam
    | RedactedThinkingBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | ServerToolUseBlockParam
    | WebSearchToolResultBlockParam
)


class MessageParam(BaseModel):
    role: MessagesRole
    content: str | ContentBlockParam


class Metadata(BaseModel):
    user_id: str | None = None


class ThinkingConfigEnabled(BaseModel):
    type: Literal["enabled"]
    budget_tokens: int


class ThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"]


type ThinkingConfigParam = ThinkingConfigEnabled | ThinkingConfigDisabled


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"]
    disable_parallel_tool_use: bool = False


class ToolChoiceAny(BaseModel):
    type: Literal["any"]
    disable_parallel_tool_use: bool = False


class ToolChoiceTool(BaseModel):
    type: Literal["tool"]
    name: str
    disable_parallel_tool_use: bool = False


class ToolChoiceNone(BaseModel):
    type: Literal["none"]


type ToolChoice = ToolChoiceAuto | ToolChoiceAny | ToolChoiceTool | ToolChoiceNone


class ToolInputSchema(BaseModel):
    type: Literal["object"]
    properties: dict[Any, Any] | None = None
    required: list[str] | None = None


class Tool(BaseModel):
    type: Literal["custom"] = "custom"
    name: str
    description: str | None = None
    input_schema: ToolInputSchema
    cache_control: CacheControlEphemeral | None = None


class ToolBash20250124(BaseModel):
    type: Literal["bash_20250124"]
    name: Literal["bash"]
    cache_control: CacheControlEphemeral | None = None


class ToolTextEditor20250124(BaseModel):
    type: Literal["text_editor_20250124"]
    name: Literal["str_replace_editor"]
    cache_control: CacheControlEphemeral | None = None


class ToolTextEditor20250429(BaseModel):
    type: Literal["text_editor_20250429"]
    name: Literal["str_replace_based_edit_tool"]
    cache_control: CacheControlEphemeral | None = None


class ToolTextEditor20250728(BaseModel):
    type: Literal["text_editor_20250728"]
    name: Literal["str_replace_based_edit_tool"]
    cache_control: CacheControlEphemeral | None = None
    max_characters: Annotated[int | None, Field(ge=1)] = None


class MessagesUserLocation(BaseModel):
    type: Literal["approximate"]
    city: str = ""
    country: str = ""
    region: str = ""
    timezone: str = ""


class WebSearchTool20250305(BaseModel):
    type: Literal["web_search_20250305"]
    name: Literal["web_search"]
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    cache_control: CacheControlEphemeral | None = None
    max_uses: int | None = None
    user_location: MessagesUserLocation | None = None


type ToolUnion = Tool | ToolBash20250124 | ToolTextEditor20250124 | ToolTextEditor20250429 | ToolTextEditor20250728 | WebSearchTool20250305


class MessagesRequest(BaseModel):
    max_tokens: Annotated[int, Field(ge=1)] = 8000
    messages: list[MessageParam]
    model: str
    metadata: Metadata | None = None
    service_tier: Literal["auto", "standard_only"] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    system: str | TextBlockParam = ""
    temperature: Annotated[float, Field(ge=0, le=1)] | None = 1.0
    thinking: ThinkingConfigParam | None = None
    tool_choice: ToolChoice | None = None
    tools: list[ToolUnion] | None = None
    top_k: Annotated[float, Field(ge=0)] | None = None
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None

    model_config = {
        "json_schema_extra": {"examples": [{"model": "qwen3:0.6b", "messages": [{"role": "user", "content": "Say Hello World o/."}]}]}
    }


class TextBlock(BaseModel):
    type: Literal["text"]
    text: str
    citations: list[TextCitationParam]


class ThinkingBlock(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"]
    data: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[Any, Any]


class ServerToolUseBlock(BaseModel):
    type: Literal["server_tool_use"]
    name: Literal["web_search"]
    id: str
    input: dict[Any, Any]


class WebSearchToolResultBlock(BaseModel):
    type: Literal["web_search_tool_result"]
    content: WebSearchToolResultBlockContent
    tool_use_id: str


type Content = TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolUseBlock


type StopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"]


class CacheCreation(BaseModel):
    ephemeral_1h_input_tokens: int
    ephemeral_5m_input_tokens: int


class ServerToolUsage(BaseModel):
    web_search_requests: int


class MessagesUsage(BaseModel):
    cache_creation: CacheCreation
    cache_creation_inpuit_tokens: int
    cache_read_input_tokens: int
    input_tokens: int
    output_tokens: int
    server_tool_use: ServerToolUsage
    service_tier: Literal["standard", "priority", "batch"]


class MessagesResponse(BaseModel):
    type: Literal["message"]
    id: str
    content: list[Content]
    model: str
    role: MessagesRole
    stop_reason: StopReason
    stop_sequence: str
    usage: MessagesUsage


RerankDocumentInput = str | dict[str, Any]


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[RerankDocumentInput]
    top_n: int | None = None
    return_documents: bool | None = None
    rank_fields: list[str] | None = None
    max_chunks_per_doc: int | None = None


RerankDocumentOutput = dict[str, Any]


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float
    document: RerankDocumentOutput | None = None


class RerankResponse(BaseModel):
    id: str | None = None
    results: list[RerankResultItem]
    meta: dict[str, Any] | None = None


SystemOneText = str | dict[str, Any] | list[Any]


class SystemOneChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["choice"]
    instructions: SystemOneText | None = None
    criteria: dict[str, str | None]


class SystemOneNoulQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["noul"]
    instructions: SystemOneText | None = None
    criteria: dict[str, str] | None = None


class SystemOneScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["score"]
    instructions: SystemOneText | None = None
    criteria: list[str]


SystemOneQuestion = Annotated[SystemOneChoiceQuestion | SystemOneNoulQuestion | SystemOneScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    """Request for the decision-model endpoint `POST /v1/systemone` (Ollama >= 0.35.0, vLLM >= 0.32.0).

    Limits such as question count, option count, score levels or request size are model- and version-dependent, so they
    are left to the backend, which rejects violations with a 400 that is passed through unchanged. Question
    `instructions` are optional and `state` may be `null`, as vLLM allows; Ollama rejects both with a 400.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    state: SystemOneText | None
    questions: dict[str, SystemOneQuestion]
    images: list[str] | None = None
    keep_alive: str | int | float | None = None

    @model_serializer(mode="wrap")
    def _keep_null_state(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep `state` in the dump even when it is `null`, since backends require the field and proxying drops `None`s."""
        data: dict[str, Any] = handler(self)
        data.setdefault("state", None)
        return data


class SystemOneChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


class SystemOneNoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class SystemOneScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


SystemOneAnswer = Annotated[SystemOneChoiceAnswer | SystemOneNoulAnswer | SystemOneScoreAnswer, Field(discriminator="type")]


class SystemOneUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, SystemOneAnswer]
    usage: SystemOneUsage
