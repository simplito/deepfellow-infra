"""Stable diffusion service."""

from server.stable_diffusion import StableDiffusionOptions, stable_diffusion

service = stable_diffusion(
    StableDiffusionOptions(
        name="automatic1",
        model_name="placeholder",
        env_vars={},
        hf_token=None,
    )
)
