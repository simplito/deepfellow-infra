from llamacpp import llamacpp, LlamacppOptions

service = llamacpp(LlamacppOptions(
        name="hermes",
        model_path="https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/blob/main/Hermes-3-Llama-3.2-3B-Q5_K_M.gguf",
        env_vars={},
        hf_token=None,
        command=""
        )
    )