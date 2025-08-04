from server.stable_diffusion import stable_diffusion, StableDiffusionOptions

service = stable_diffusion(StableDiffusionOptions(
        name="automatic1",
        model_name="placeholder",
        env_vars={},
        hf_token=None,
    )
)