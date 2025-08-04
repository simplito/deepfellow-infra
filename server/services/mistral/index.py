from server.llamacpp import llamacpp, LlamacppOptions

service = llamacpp(LlamacppOptions(
        name="mistral",
        model_path="https://huggingface.co/bartowski/mistral-community_pixtral-12b-GGUF/blob/main/mistral-community_pixtral-12b-Q5_K_M.gguf",
        env_vars={},
        hf_token=None,
        command=""
        )
    )