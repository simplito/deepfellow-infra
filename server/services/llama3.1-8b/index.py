from server.llamacpp import llamacpp, LlamacppOptions

service = llamacpp(LlamacppOptions(
        name="llama3.1-8b-instruct",
        model_path="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        env_vars={},
        hf_token=None,
        command=""
        )
    )