"""Mistral service."""

from server.llamacpp import LlamacppOptions, llamacpp

service = llamacpp(
    LlamacppOptions(
        name="mistral",
        model_path="https://huggingface.co/bartowski/mistral-community_pixtral-12b-GGUF/blob/main/mistral-community_pixtral-12b-Q5_K_M.gguf",
        env_vars={},
    )
)
