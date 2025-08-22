"""Coqui service."""

from server.coqui import CoquiOptions, coqui

service = coqui(
    CoquiOptions(
        name="tts",
        model_name="tts_models/en/vctk/vits",
        default_speaker="p225",
    )
)
