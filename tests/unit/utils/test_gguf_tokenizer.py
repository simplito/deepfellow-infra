# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/gguf_tokenizer.py.

The vocabularies are tiny stand-ins for real GGUF ones, spelled the way llama.cpp stores them. Token counts on real
models were checked against Ollama's own `prompt_eval_count` for every embedding family in the catalog.
"""

import base64
import json
from typing import Any

import pytest
from tokenizers import Tokenizer

from server.utils.gguf_tokenizer import _precompiled_charsmap, build_gguf_tokenizer, tokenizer_files  # pyright: ignore[reportPrivateUsage]

NORMAL, CONTROL, USER_DEFINED = 1, 3, 4


def _info(kind: str, tokens: list[str], types: list[int] | None = None, **fields: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "general.architecture": "test",
        "tokenizer.ggml.model": kind,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types if types is not None else [NORMAL] * len(tokens),
    }
    info.update({f"tokenizer.ggml.{key}": value for key, value in fields.items()})
    return info


def _tokens(tokenizer: Tokenizer | None, text: str) -> list[str]:
    assert tokenizer is not None
    return tokenizer.encode(text).tokens


# ─── bert (WordPiece) ─────────────────────────────────────────────────────────

_BERT_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "▁hello", "▁world", "▁play", "ing", "▁.", "▁ᄒ", "ᄀ", "[unused0]"]
_BERT_TYPES = [CONTROL] * 5 + [NORMAL] * 8


def _bert(**fields: Any) -> dict[str, Any]:
    ids = {"unknown_token_id": 1, "padding_token_id": 0, "cls_token_id": 2, "seperator_token_id": 3, "mask_token_id": 4}
    return _info("bert", _BERT_TOKENS, _BERT_TYPES, **(ids | fields))


def test_bert_respells_llama_cpp_word_starts_and_wraps_in_cls_sep() -> None:
    tokenizer = build_gguf_tokenizer(_bert())

    assert _tokens(tokenizer, "Hello world. Playing") == ["[CLS]", "hello", "world", ".", "play", "##ing", "[SEP]"]


def test_bert_maps_hangul_syllables_to_their_leading_consonant_like_llama_cpp() -> None:
    tokenizer = build_gguf_tokenizer(_bert())

    assert _tokens(tokenizer, "한국") == ["[CLS]", "ᄒ", "##ᄀ", "[SEP]"]


def test_bert_strips_accents() -> None:
    tokenizer = build_gguf_tokenizer(
        _info(
            "bert", ["[UNK]", "[CLS]", "[SEP]", "▁cafe"], [CONTROL] * 3 + [NORMAL], unknown_token_id=0, cls_token_id=1, seperator_token_id=2
        )
    )

    assert _tokens(tokenizer, "Café") == ["[CLS]", "cafe", "[SEP]"]


def test_bert_splits_a_word_of_any_length_into_pieces_like_llama_cpp() -> None:
    tokenizer = build_gguf_tokenizer(
        _info(
            "bert",
            ["[UNK]", "[CLS]", "[SEP]", "▁1", "1"],
            [CONTROL] * 3 + [NORMAL] * 2,
            unknown_token_id=0,
            cls_token_id=1,
            seperator_token_id=2,
        )
    )

    assert _tokens(tokenizer, "1" * 150) == ["[CLS]", "1", *["##1"] * 149, "[SEP]"]


def test_bert_splits_cjk_characters_into_words() -> None:
    tokenizer = build_gguf_tokenizer(
        _info(
            "bert",
            ["[UNK]", "[CLS]", "[SEP]", "▁中", "▁文"],
            [CONTROL] * 3 + [NORMAL] * 2,
            unknown_token_id=0,
            cls_token_id=1,
            seperator_token_id=2,
        )
    )

    assert _tokens(tokenizer, "中文") == ["[CLS]", "中", "文", "[SEP]"]


def test_bert_falls_back_to_bos_and_eos_without_cls_and_sep_ids() -> None:
    tokenizer = build_gguf_tokenizer(_info("bert", _BERT_TOKENS, _BERT_TYPES, unknown_token_id=1, bos_token_id=2, eos_token_id=3))

    assert _tokens(tokenizer, "hello") == ["[CLS]", "hello", "[SEP]"]


def test_bert_without_special_ids_adds_nothing_around_the_text() -> None:
    tokenizer = build_gguf_tokenizer(_info("bert", _BERT_TOKENS, _BERT_TYPES, unknown_token_id=1))

    assert _tokens(tokenizer, "hello") == ["hello"]


# ─── gpt2 (byte-level BPE) ────────────────────────────────────────────────────


def _gpt2(pre: str, **fields: Any) -> dict[str, Any]:
    tokens = ["h", "i", "Ġ", "hi", "Ġhi", "<|endoftext|>", "'", "s"]
    return _info("gpt2", tokens, [NORMAL] * 5 + [CONTROL, NORMAL, NORMAL], pre=pre, merges=["h i", "Ġ hi"], eos_token_id=5, **fields)


def test_gpt2_qwen2_appends_eos_when_the_file_says_so() -> None:
    tokenizer = build_gguf_tokenizer(_gpt2("qwen2", add_eos_token=True))

    assert _tokens(tokenizer, "hi hi") == ["hi", "Ġhi", "<|endoftext|>"]


def test_gpt2_adds_nothing_by_default() -> None:
    tokenizer = build_gguf_tokenizer(_gpt2("gpt-2"))

    assert _tokens(tokenizer, "hi hi") == ["hi", "Ġhi"]


def test_gpt2_matches_control_tokens_whole() -> None:
    tokenizer = build_gguf_tokenizer(_gpt2("qwen2"))

    assert _tokens(tokenizer, "hi<|endoftext|>") == ["hi", "<|endoftext|>"]


@pytest.mark.parametrize(
    ("pre", "expected"),
    [("gpt-2", ["123", "Ġ!", "a"]), ("qwen2", ["1", "2", "3", "Ġ!", "a"]), (None, ["123", "Ġ!", "a"])],
)
def test_gpt2_splits_text_with_the_pattern_of_its_pre_tokenizer(pre: str | None, expected: list[str]) -> None:
    fields: dict[str, Any] = {"merges": ["1 2", "12 3", "Ġ !"]} | ({"pre": pre} if pre else {})
    tokenizer = build_gguf_tokenizer(_info("gpt2", ["1", "2", "3", "12", "123", "Ġ", "!", "Ġ!", "a"], **fields))

    assert _tokens(tokenizer, "123 !a") == expected


def test_gpt2_with_unknown_pre_tokenizer_has_no_tokenizer() -> None:
    assert build_gguf_tokenizer(_gpt2("llama-bpe")) is None


# ─── t5 (Unigram, XLM-R style) ────────────────────────────────────────────────

_T5_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "▁hello", "▁world", "▁", "h"]
_T5_SCORES = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -2.0, -3.0]


def test_t5_splits_on_whitespace_and_wraps_in_bos_eos() -> None:
    info = _info("t5", _T5_TOKENS, [CONTROL] * 4 + [NORMAL] * 4, scores=_T5_SCORES, unknown_token_id=3, bos_token_id=0, eos_token_id=2)

    assert _tokens(build_gguf_tokenizer(info), "hello   world") == ["<s>", "▁hello", "▁world", "</s>"]


def test_t5_with_a_broken_charsmap_raises() -> None:
    info = _info("t5", _T5_TOKENS, scores=_T5_SCORES, unknown_token_id=3, precompiled_charsmap=base64.b64encode(b"junk").decode())

    with pytest.raises(Exception):  # noqa: B017, PT011
        build_gguf_tokenizer(info)


def test_precompiled_charsmap_reads_every_encoding_files_use() -> None:
    raw = b"\x01\x02charsmap"
    encoded = base64.b64encode(raw).decode()

    assert _precompiled_charsmap(encoded) == raw
    assert _precompiled_charsmap(list(encoded)) == raw
    assert _precompiled_charsmap(list(raw)) == raw
    assert _precompiled_charsmap([-1, 1]) == b"\xff\x01"
    assert _precompiled_charsmap(None) is None
    assert _precompiled_charsmap("") is None


# ─── llama (SentencePiece BPE, embeddinggemma) ────────────────────────────────

_SPM_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>", "h", "i", "▁", "hi", "▁hi", "<0x41>"]
_SPM_SCORES = [0.0, 0.0, 0.0, 0.0, -5.0, -5.0, -5.0, -1.0, -2.0, 0.0]


def _spm(kind: str, **fields: Any) -> dict[str, Any]:
    return _info(
        kind,
        _SPM_TOKENS,
        [CONTROL] * 4 + [NORMAL] * 5 + [6],
        scores=_SPM_SCORES,
        unknown_token_id=3,
        bos_token_id=2,
        eos_token_id=1,
        **fields,
    )


def test_llama_derives_merges_from_scores_and_adds_bos() -> None:
    tokenizer = build_gguf_tokenizer(_spm("llama", add_bos_token=True, add_eos_token=False))

    assert _tokens(tokenizer, "hi hi") == ["<bos>", "hi", "▁hi"]


def test_llama_merges_the_pair_with_the_higher_score_first() -> None:
    info = _info(
        "llama",
        ["<unk>", "a", "b", "c", "ab", "bc"],
        [CONTROL] + [NORMAL] * 5,
        scores=[0.0, -5.0, -5.0, -5.0, -2.0, -1.0],
        unknown_token_id=0,
    )

    assert _tokens(build_gguf_tokenizer(info | {"tokenizer.ggml.add_bos_token": False}), "abc") == ["a", "bc"]


def test_llama_falls_back_to_bytes_for_unknown_characters() -> None:
    tokenizer = build_gguf_tokenizer(_spm("llama", add_bos_token=False))

    assert _tokens(tokenizer, "A") == ["<0x41>"]


def test_llama_prepends_a_space_when_the_file_says_so() -> None:
    tokenizer = build_gguf_tokenizer(_spm("llama", add_bos_token=False, add_space_prefix=True))

    assert _tokens(tokenizer, "hi") == ["▁hi"]


# ─── gemma4 (SentencePiece-style BPE with stored merges, embeddinggemma-2) ────


def test_gemma4_uses_stored_merges_and_always_adds_bos() -> None:
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "h", "i", "▁", "hi", "▁hi", "\n", "\n\n"]
    info = _info(
        "gemma4",
        tokens,
        [CONTROL] * 4 + [NORMAL] * 7,
        merges=["h i", "▁ hi", "\n \n"],
        unknown_token_id=3,
        bos_token_id=2,
        eos_token_id=1,
        add_bos_token=False,
        add_eos_token=True,
    )

    assert _tokens(build_gguf_tokenizer(info), "hi hi\n\nhi") == ["<bos>", "hi", "▁hi", "\n\n", "hi", "<eos>"]


def test_gemma4_never_merges_across_a_newline() -> None:
    info = _info(
        "gemma4",
        ["<pad>", "<eos>", "<bos>", "<unk>", "h", "i", "hi", "\n", "i\n"],
        [CONTROL] * 4 + [NORMAL] * 5,
        merges=["i \n", "h i"],
        unknown_token_id=3,
        bos_token_id=2,
    )

    assert _tokens(build_gguf_tokenizer(info), "hi\nhi") == ["<bos>", "hi", "\n", "hi"]


def test_gemma4_falls_back_to_bytes_for_unknown_characters() -> None:
    info = _info(
        "gemma4", ["<pad>", "<eos>", "<bos>", "<unk>", "<0x41>"], [CONTROL] * 4 + [6], merges=[], unknown_token_id=3, bos_token_id=2
    )

    assert _tokens(build_gguf_tokenizer(info), "A") == ["<bos>", "<0x41>"]


# ─── shared ───────────────────────────────────────────────────────────────────


def test_user_defined_tokens_match_before_normalization() -> None:
    tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "h", "i", "▁", "hi", "<start_of_turn>"]
    info = _info(
        "llama",
        tokens,
        [CONTROL] * 4 + [NORMAL] * 4 + [USER_DEFINED],
        scores=[0.0] * 4 + [-5.0, -5.0, -5.0, -1.0, 0.0],
        unknown_token_id=3,
        add_bos_token=False,
    )

    assert _tokens(build_gguf_tokenizer(info), "<start_of_turn>hi") == ["<start_of_turn>", "hi"]


def test_unsupported_tokenizer_type_has_no_tokenizer() -> None:
    assert build_gguf_tokenizer(_info("rwkv", ["a"])) is None


def test_model_without_tokens_has_no_tokenizer() -> None:
    assert build_gguf_tokenizer({"tokenizer.ggml.model": "bert"}) is None


def test_mismatched_token_types_are_treated_as_normal() -> None:
    tokenizer = build_gguf_tokenizer(_info("bert", ["▁hello", "[UNK]"], [NORMAL], unknown_token_id=1))

    assert _tokens(tokenizer, "hello") == ["hello"]


def test_tokenizer_files_name_special_tokens_for_auto_tokenizer() -> None:
    info = _bert()
    tokenizer = build_gguf_tokenizer(info)
    assert tokenizer is not None

    files = tokenizer_files(info, tokenizer)

    assert json.loads(files["tokenizer_config.json"]) == {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }
    assert Tokenizer.from_str(files["tokenizer.json"].decode()).encode("hello").tokens == ["[CLS]", "hello", "[SEP]"]


@pytest.mark.parametrize("kind", ["t5", "gemma4"])
def test_without_special_token_ids_nothing_wraps_the_text(kind: str) -> None:
    info = _info(
        kind,
        ["<unk>", "hi", "▁hi", "h", "i", "▁"],
        [CONTROL] + [NORMAL] * 5,
        scores=[0.0, -1.0, -1.0, -2.0, -2.0, -2.0],
        merges=["h i"],
        unknown_token_id=0,
    )

    assert len(_tokens(build_gguf_tokenizer(info), "hi")) == 1


def test_bos_that_is_also_eos_is_added_at_both_ends() -> None:
    info = _gpt2("gpt-2", bos_token_id=5, add_bos_token=True, add_eos_token=True)

    assert _tokens(build_gguf_tokenizer(info), "hi") == ["<|endoftext|>", "hi", "<|endoftext|>"]


def test_token_types_that_are_not_numbers_raise() -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        build_gguf_tokenizer(_info("bert", ["▁hello", "[UNK]"], ["x", NORMAL], unknown_token_id=1))  # pyright: ignore[reportArgumentType]


def _wrapped(kind: str, **flags: bool) -> tuple[dict[str, Any], str, list[str], str, str]:
    if kind == "bert":
        return _bert(**flags), "hello", ["hello"], "[CLS]", "[SEP]"
    if kind == "t5":
        info = _info(
            "t5", _T5_TOKENS, [CONTROL] * 4 + [NORMAL] * 4, scores=_T5_SCORES, unknown_token_id=3, bos_token_id=0, eos_token_id=2, **flags
        )
        return info, "hello", ["▁hello"], "<s>", "</s>"
    return _spm("llama", **flags), "hi", ["hi"], "<bos>", "<eos>"


@pytest.mark.parametrize(
    ("kind", "flags", "bos", "eos"),
    [
        ("bert", {}, True, True),
        ("bert", {"add_bos_token": False, "add_eos_token": False}, False, False),
        ("bert", {"add_bos_token": True, "add_eos_token": True}, True, True),
        ("t5", {}, True, True),
        ("t5", {"add_bos_token": False, "add_eos_token": False}, False, False),
        ("t5", {"add_bos_token": True, "add_eos_token": True}, True, True),
        ("llama", {}, True, False),
        ("llama", {"add_bos_token": False, "add_eos_token": False}, False, False),
        ("llama", {"add_bos_token": True, "add_eos_token": True}, True, True),
    ],
)
def test_bos_and_eos_follow_the_file_flags_and_their_defaults(kind: str, flags: dict[str, bool], bos: bool, eos: bool) -> None:
    info, text, body, bos_token, eos_token = _wrapped(kind, **flags)

    assert _tokens(build_gguf_tokenizer(info), text) == [*([bos_token] if bos else []), *body, *([eos_token] if eos else [])]
