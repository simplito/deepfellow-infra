# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Rebuild the tokenizer an Ollama model carries in its GGUF file as a Hugging Face tokenizer that counts like Ollama."""

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

from server.utils.logger import logger

_NORMAL = 1
_CONTROL = 3
_USER_DEFINED = 4

_SPIECE_UNDERLINE = "▁"
_NO_WORD_LENGTH_LIMIT = 1_000_000

_GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_QWEN2_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
_BPE_SPLIT_PATTERNS = {"gpt-2": _GPT2_PATTERN, "default": _GPT2_PATTERN, "qwen2": _QWEN2_PATTERN}


@dataclass
class _Gguf:
    """The `tokenizer.ggml.*` fields of a model's `model_info`."""

    info: Mapping[str, object]

    def get(self, key: str) -> object:
        return self.info.get(f"tokenizer.ggml.{key}")

    def strings(self, key: str) -> list[str]:
        value = self.get(key)
        return [str(v) for v in value] if isinstance(value, list) else []

    def token_id(self, key: str) -> int | None:
        value = self.get(key)
        return value if isinstance(value, int) and value >= 0 else None

    def flag(self, key: str, default: bool) -> bool:
        value = self.get(key)
        return value if isinstance(value, bool) else default


def build_gguf_tokenizer(model_info: Mapping[str, object]) -> Tokenizer | None:
    """Return the tokenizer stored in `model_info` from `/api/show` verbose, or None for an unsupported one; raise if it is broken."""
    gguf = _Gguf(model_info)
    kind = gguf.get("model")
    tokens = gguf.strings("tokens")
    if not tokens:
        return None
    types = gguf.get("token_type")
    token_types = [int(t) for t in types] if isinstance(types, list) and len(types) == len(tokens) else [_NORMAL] * len(tokens)
    if kind == "bert":
        tokenizer = _wordpiece(gguf, tokens, token_types)
    elif kind == "t5":
        tokenizer = _unigram(gguf, tokens)
    elif kind == "gpt2":
        tokenizer = _byte_level_bpe(gguf, tokens)
    elif kind == "llama":
        tokenizer = _sentencepiece_bpe(gguf, tokens)
    elif kind == "gemma4":
        tokenizer = _gemma4_bpe(gguf, tokens)
    else:
        logger.info("No tokenizer conversion for GGUF tokenizer type %r", kind)
        return None
    if tokenizer is not None:
        _add_special_tokens(tokenizer, tokens, token_types)
    return tokenizer


def tokenizer_files(model_info: Mapping[str, object], tokenizer: Tokenizer) -> dict[str, bytes]:
    """Return `tokenizer.json` and a `tokenizer_config.json` that lets `AutoTokenizer` load it."""
    gguf = _Gguf(model_info)
    tokens = gguf.strings("tokens")
    config: dict[str, object] = {"tokenizer_class": "PreTrainedTokenizerFast", "clean_up_tokenization_spaces": False}
    for name, key in (
        ("bos_token", "bos_token_id"),
        ("eos_token", "eos_token_id"),
        ("unk_token", "unknown_token_id"),
        ("pad_token", "padding_token_id"),
        ("cls_token", "cls_token_id"),
        ("sep_token", "seperator_token_id"),
        ("mask_token", "mask_token_id"),
    ):
        token_id = gguf.token_id(key)
        if token_id is not None and token_id < len(tokens):
            config[name] = tokens[token_id]
    return {
        "tokenizer.json": tokenizer.to_str().encode(),
        "tokenizer_config.json": json.dumps(config, indent=2, ensure_ascii=False).encode(),
    }


def _add_special_tokens(tokenizer: Tokenizer, tokens: Sequence[str], token_types: Sequence[int]) -> None:
    """Match control tokens whole, as llama.cpp does, and user-defined ones before normalization."""
    control = [AddedToken(t, special=True, normalized=False) for t, kind in zip(tokens, token_types, strict=True) if kind == _CONTROL]
    user_defined = [
        AddedToken(t, special=False, normalized=False) for t, kind in zip(tokens, token_types, strict=True) if kind == _USER_DEFINED
    ]
    if control:
        tokenizer.add_special_tokens(control)
    if user_defined:
        tokenizer.add_tokens(user_defined)


def _template(
    gguf: _Gguf, tokens: Sequence[str], bos_key: str, eos_key: str, add_bos: bool, add_eos: bool
) -> processors.TemplateProcessing | None:
    """Wrap the text in the special tokens Ollama adds around it."""
    bos_id = gguf.token_id(bos_key) if add_bos else None
    eos_id = gguf.token_id(eos_key) if add_eos else None
    if bos_id is None and eos_id is None:
        return None
    bos = [tokens[bos_id]] if bos_id is not None else []
    eos = [tokens[eos_id]] if eos_id is not None else []
    specials = {tokens[token_id]: token_id for token_id in (bos_id, eos_id) if token_id is not None}
    return processors.TemplateProcessing(
        single=[*bos, "$A", *eos],
        pair=[*bos, "$A", *eos, "$B:1", *(f"{token}:1" for token in eos)],
        special_tokens=list(specials.items()),
    )


def _wordpiece(gguf: _Gguf, tokens: Sequence[str], token_types: Sequence[int]) -> Tokenizer:
    """WordPiece back from llama.cpp's spelling: a word start is `▁word` there, `word` in HF; a continuation is `##`-less there."""
    vocab: dict[str, int] = {}
    for index, (token, kind) in enumerate(zip(tokens, token_types, strict=True)):
        if kind == _CONTROL or (token.startswith("[") and token.endswith("]") and len(token) > 1):
            word = token
        elif token.startswith(_SPIECE_UNDERLINE):
            word = token[1:]
        else:
            word = f"##{token}"
        vocab.setdefault(word, index)
    unk_id = gguf.token_id("unknown_token_id")
    unk = tokens[unk_id] if unk_id is not None else "[UNK]"
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token=unk, max_input_chars_per_word=_NO_WORD_LENGTH_LIMIT))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=False, lowercase=True),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Replace(Regex("[\u1160-\u11ff]"), ""),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    cls_key = "cls_token_id" if gguf.token_id("cls_token_id") is not None else "bos_token_id"
    sep_key = "seperator_token_id" if gguf.token_id("seperator_token_id") is not None else "eos_token_id"
    template = _template(gguf, tokens, cls_key, sep_key, gguf.flag("add_bos_token", True), gguf.flag("add_eos_token", True))
    if template is not None:
        tokenizer.post_processor = template
    return tokenizer


def _precompiled_charsmap(value: object) -> bytes | None:
    """Read `precompiled_charsmap`: a base64 string, a list of its characters, or a list of the raw (maybe signed) bytes."""
    if isinstance(value, list) and value:
        if all(isinstance(v, int) for v in value):
            return bytes(v & 0xFF for v in value)  # pyright: ignore[reportOperatorIssue]
        value = "".join(str(v) for v in value)
    if isinstance(value, str) and value:
        return base64.b64decode(value)
    return None


def _unigram(gguf: _Gguf, tokens: Sequence[str]) -> Tokenizer:
    scores_value = gguf.get("scores")
    scores = (
        [float(s) for s in scores_value] if isinstance(scores_value, list) and len(scores_value) == len(tokens) else [0.0] * len(tokens)
    )
    unk_id = gguf.token_id("unknown_token_id")
    tokenizer = Tokenizer(models.Unigram(list(zip(tokens, scores, strict=True)), unk_id=unk_id, byte_fallback=False))
    steps: list[normalizers.Normalizer] = []
    charsmap = _precompiled_charsmap(gguf.get("precompiled_charsmap"))
    if charsmap:
        steps.append(normalizers.Precompiled(charsmap))
    steps.append(normalizers.Replace(Regex(" {2,}"), " "))
    tokenizer.normalizer = normalizers.Sequence(steps)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace(replacement=_SPIECE_UNDERLINE, prepend_scheme="always")]
    )
    tokenizer.decoder = decoders.Metaspace(replacement=_SPIECE_UNDERLINE, prepend_scheme="always")
    template = _template(gguf, tokens, "bos_token_id", "eos_token_id", gguf.flag("add_bos_token", True), gguf.flag("add_eos_token", True))
    if template is not None:
        tokenizer.post_processor = template
    return tokenizer


def _byte_level_bpe(gguf: _Gguf, tokens: Sequence[str]) -> Tokenizer | None:
    pre = gguf.get("pre")
    pattern = _BPE_SPLIT_PATTERNS.get(pre if isinstance(pre, str) else "default")
    if pattern is None:
        logger.info("No pre-tokenizer for GGUF BPE variant %r", pre)
        return None
    vocab = {token: index for index, token in enumerate(tokens)}
    merges = [tuple(m.split(" ", 1)) for m in gguf.strings("merges") if " " in m]
    tokenizer = Tokenizer(models.BPE(vocab, merges, fuse_unk=False, byte_fallback=False))  # pyright: ignore[reportArgumentType]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    template = _template(gguf, tokens, "bos_token_id", "eos_token_id", gguf.flag("add_bos_token", False), gguf.flag("add_eos_token", False))
    tokenizer.post_processor = template if template is not None else processors.ByteLevel(trim_offsets=False)
    return tokenizer


def _spm_merges(tokens: Sequence[str], scores: Sequence[float]) -> list[tuple[str, str]]:
    """Derive BPE merges from SentencePiece scores, as transformers' SentencePieceExtractor does."""
    vocab = {token: index for index, token in enumerate(tokens)}
    merges: list[tuple[str, str, float]] = []
    for piece, score in zip(tokens, scores, strict=True):
        local = [(piece[:i], piece[i:], score) for i in range(1, len(piece)) if piece[:i] in vocab and piece[i:] in vocab]
        local.sort(key=lambda m: (vocab[m[0]], vocab[m[1]]))
        merges.extend(local)
    merges.sort(key=lambda m: m[2], reverse=True)
    return [(left, right) for left, right, _ in merges]


def _sentencepiece_bpe(gguf: _Gguf, tokens: Sequence[str]) -> Tokenizer:
    scores_value = gguf.get("scores")
    scores = (
        [float(s) for s in scores_value] if isinstance(scores_value, list) and len(scores_value) == len(tokens) else [0.0] * len(tokens)
    )
    vocab = {token: index for index, token in enumerate(tokens)}
    unk_id = gguf.token_id("unknown_token_id")
    tokenizer = Tokenizer(
        models.BPE(
            vocab, _spm_merges(tokens, scores), unk_token=tokens[unk_id] if unk_id is not None else None, fuse_unk=True, byte_fallback=True
        )
    )
    steps: list[normalizers.Normalizer] = []
    if gguf.flag("add_space_prefix", False):
        steps.append(normalizers.Prepend(_SPIECE_UNDERLINE))
    steps.append(normalizers.Replace(" ", _SPIECE_UNDERLINE))
    tokenizer.normalizer = normalizers.Sequence(steps)
    tokenizer.decoder = decoders.Sequence([decoders.Replace(_SPIECE_UNDERLINE, " "), decoders.ByteFallback(), decoders.Fuse()])
    template = _template(gguf, tokens, "bos_token_id", "eos_token_id", gguf.flag("add_bos_token", True), gguf.flag("add_eos_token", False))
    if template is not None:
        tokenizer.post_processor = template
    return tokenizer


def _gemma4_bpe(gguf: _Gguf, tokens: Sequence[str]) -> Tokenizer:
    """SentencePiece-style BPE with stored merges, split only at newlines and always starting with BOS, as llama.cpp's Gemma4."""
    vocab = {token: index for index, token in enumerate(tokens)}
    merges = [tuple(m.split(" ", 1)) for m in gguf.strings("merges") if " " in m]
    unk_id = gguf.token_id("unknown_token_id")
    tokenizer = Tokenizer(
        models.BPE(vocab, merges, unk_token=tokens[unk_id] if unk_id is not None else None, fuse_unk=True, byte_fallback=True)  # pyright: ignore[reportArgumentType]
    )
    tokenizer.normalizer = normalizers.Replace(" ", _SPIECE_UNDERLINE)
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("\n+"), behavior="isolated", invert=False)
    tokenizer.decoder = decoders.Sequence([decoders.Replace(_SPIECE_UNDERLINE, " "), decoders.ByteFallback(), decoders.Fuse()])
    template = _template(gguf, tokens, "bos_token_id", "eos_token_id", True, gguf.flag("add_eos_token", False))
    if template is not None:
        tokenizer.post_processor = template
    return tokenizer
