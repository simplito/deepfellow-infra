"""PLLuM service."""

from llamacpp import LlamacppOptions, llamacpp

service = llamacpp(
    LlamacppOptions(
        name="pllum",
        model_path="https://huggingface.co/mradermacher/PLLuM-12B-instruct-GGUF/resolve/main/PLLuM-12B-instruct.Q4_K_M.gguf",
        env_vars={},
    )
)
