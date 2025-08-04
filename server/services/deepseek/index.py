from server.llamacpp import llamacpp, LlamacppOptions

service = llamacpp(LlamacppOptions(
        name="deepseek-r1",
        model_path="https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-GGUF/blob/main/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B-Q5_K_M.gguf",
        env_vars={},
        hf_token=None,
        command=""
        )
    )