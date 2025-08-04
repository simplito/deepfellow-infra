from server.coqui import coqui, CoquiOptions

service = coqui(CoquiOptions(
        name="tts",
        model_name="tts_models/en/vctk/vits",
        default_speaker="p225",
    )
)