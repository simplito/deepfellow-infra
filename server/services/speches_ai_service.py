"""Speaches AI service."""

from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel

from server.docker import DockerOptions, install_and_run_docker, uninstall_docker
from server.endpointregistry import ProxyOptions
from server.models.models import InstallModelIn, ListModelsFilters, ListModelsOut, RetrieveModelOut, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import fetch_from_localhost

ModelType = Literal["tts", "stt"]


class SpeachesAiConst(BaseModel):
    image_gpu: str
    image_cpu: str
    audio_speech_models: list[str]
    audio_transcriptions_models: list[str]
    models: dict[str, ModelType]


_const = SpeachesAiConst(
    image_gpu="ghcr.io/speaches-ai/speaches:latest",
    image_cpu="ghcr.io/speaches-ai/speaches:latest-cpu",
    audio_speech_models=[
        "speaches-ai/Kokoro-82M-v1.0-ONNX-fp16",
        "speaches-ai/Kokoro-82M-v1.0-ONNX",
        "speaches-ai/Kokoro-82M-v1.0-ONNX-int8",
        "speaches-ai/piper-ar_JO-kareem-low",
        "speaches-ai/piper-ar_JO-kareem-medium",
        "speaches-ai/piper-ca_ES-upc_ona-medium",
        "speaches-ai/piper-ca_ES-upc_ona-x_low",
        "speaches-ai/piper-ca_ES-upc_pau-x_low",
        "speaches-ai/piper-cs_CZ-jirka-low",
        "speaches-ai/piper-cs_CZ-jirka-medium",
        "speaches-ai/piper-cy_GB-gwryw_gogleddol-medium",
        "speaches-ai/piper-da_DK-talesyntese-medium",
        "speaches-ai/piper-de_DE-eva_k-x_low",
        "speaches-ai/piper-de_DE-karlsson-low",
        "speaches-ai/piper-de_DE-kerstin-low",
        "speaches-ai/piper-de_DE-mls-medium",
        "speaches-ai/piper-de_DE-pavoque-low",
        "speaches-ai/piper-de_DE-ramona-low",
        "speaches-ai/piper-de_DE-thorsten-high",
        "speaches-ai/piper-de_DE-thorsten-low",
        "speaches-ai/piper-de_DE-thorsten-medium",
        "speaches-ai/piper-de_DE-thorsten_emotional-medium",
        "speaches-ai/piper-el_GR-rapunzelina-low",
        "speaches-ai/piper-en_GB-alan-low",
        "speaches-ai/piper-en_GB-alan-medium",
        "speaches-ai/piper-en_GB-alba-medium",
        "speaches-ai/piper-en_GB-aru-medium",
        "speaches-ai/piper-en_GB-cori-high",
        "speaches-ai/piper-en_GB-cori-medium",
        "speaches-ai/piper-en_GB-jenny_dioco-medium",
        "speaches-ai/piper-en_GB-northern_english_male-medium",
        "speaches-ai/piper-en_GB-semaine-medium",
        "speaches-ai/piper-en_GB-southern_english_female-low",
        "speaches-ai/piper-en_GB-vctk-medium",
        "speaches-ai/piper-en_US-amy-low",
        "speaches-ai/piper-en_US-amy-medium",
        "speaches-ai/piper-en_US-arctic-medium",
        "speaches-ai/piper-en_US-bryce-medium",
        "speaches-ai/piper-en_US-danny-low",
        "speaches-ai/piper-en_US-hfc_female-medium",
        "speaches-ai/piper-en_US-hfc_male-medium",
        "speaches-ai/piper-en_US-joe-medium",
        "speaches-ai/piper-en_US-john-medium",
        "speaches-ai/piper-en_US-kathleen-low",
        "speaches-ai/piper-en_US-kristin-medium",
        "speaches-ai/piper-en_US-kusal-medium",
        "speaches-ai/piper-en_US-l2arctic-medium",
        "speaches-ai/piper-en_US-lessac-high",
        "speaches-ai/piper-en_US-lessac-low",
        "speaches-ai/piper-en_US-lessac-medium",
        "speaches-ai/piper-en_US-libritts-high",
        "speaches-ai/piper-en_US-libritts_r-medium",
        "speaches-ai/piper-en_US-ljspeech-high",
        "speaches-ai/piper-en_US-ljspeech-medium",
        "speaches-ai/piper-en_US-norman-medium",
        "speaches-ai/piper-en_US-ryan-high",
        "speaches-ai/piper-en_US-ryan-low",
        "speaches-ai/piper-en_US-ryan-medium",
        "speaches-ai/piper-es_ES-carlfm-x_low",
        "speaches-ai/piper-es_ES-davefx-medium",
        "speaches-ai/piper-es_ES-mls_10246-low",
        "speaches-ai/piper-es_ES-mls_9972-low",
        "speaches-ai/piper-es_ES-sharvard-medium",
        "speaches-ai/piper-es_MX-ald-medium",
        "speaches-ai/piper-es_MX-claude-high",
        "speaches-ai/piper-fa_IR-amir-medium",
        "speaches-ai/piper-fa_IR-gyro-medium",
        "speaches-ai/piper-fi_FI-harri-low",
        "speaches-ai/piper-fi_FI-harri-medium",
        "speaches-ai/piper-fr_FR-gilles-low",
        "speaches-ai/piper-fr_FR-mls-medium",
        "speaches-ai/piper-fr_FR-mls_1840-low",
        "speaches-ai/piper-fr_FR-siwis-low",
        "speaches-ai/piper-fr_FR-siwis-medium",
        "speaches-ai/piper-fr_FR-tom-medium",
        "speaches-ai/piper-fr_FR-upmc-medium",
        "speaches-ai/piper-hu_HU-anna-medium",
        "speaches-ai/piper-hu_HU-berta-medium",
        "speaches-ai/piper-hu_HU-imre-medium",
        "speaches-ai/piper-is_IS-bui-medium",
        "speaches-ai/piper-is_IS-salka-medium",
        "speaches-ai/piper-is_IS-steinn-medium",
        "speaches-ai/piper-is_IS-ugla-medium",
        "speaches-ai/piper-it_IT-paola-medium",
        "speaches-ai/piper-it_IT-riccardo-x_low",
        "speaches-ai/piper-ka_GE-natia-medium",
        "speaches-ai/piper-kk_KZ-iseke-x_low",
        "speaches-ai/piper-kk_KZ-issai-high",
        "speaches-ai/piper-kk_KZ-raya-x_low",
        "speaches-ai/piper-lb_LU-marylux-medium",
        "speaches-ai/piper-lv_LV-aivars-medium",
        "speaches-ai/piper-ne_NP-google-medium",
        "speaches-ai/piper-ne_NP-google-x_low",
        "speaches-ai/piper-nl_BE-nathalie-medium",
        "speaches-ai/piper-nl_BE-nathalie-x_low",
        "speaches-ai/piper-nl_BE-rdh-medium",
        "speaches-ai/piper-nl_BE-rdh-x_low",
        "speaches-ai/piper-nl_NL-mls-medium",
        "speaches-ai/piper-nl_NL-mls_5809-low",
        "speaches-ai/piper-nl_NL-mls_7432-low",
        "speaches-ai/piper-no_NO-talesyntese-medium",
        "speaches-ai/piper-pl_PL-darkman-medium",
        "speaches-ai/piper-pl_PL-gosia-medium",
        "speaches-ai/piper-pl_PL-mc_speech-medium",
        "speaches-ai/piper-pl_PL-mls_6892-low",
        "speaches-ai/piper-pt_BR-edresson-low",
        "speaches-ai/piper-pt_BR-faber-medium",
        "speaches-ai/piper-zh_CN-huayan-x_low",
        "speaches-ai/piper-ro_RO-mihai-medium",
        "speaches-ai/piper-ru_RU-denis-medium",
        "speaches-ai/piper-ru_RU-dmitri-medium",
        "speaches-ai/piper-ru_RU-irina-medium",
        "speaches-ai/piper-ru_RU-ruslan-medium",
        "speaches-ai/piper-sk_SK-lili-medium",
        "speaches-ai/piper-sl_SI-artur-medium",
        "speaches-ai/piper-sr_RS-serbski_institut-medium",
        "speaches-ai/piper-sv_SE-nst-medium",
        "speaches-ai/piper-sw_CD-lanfrica-medium",
        "speaches-ai/piper-tr_TR-dfki-medium",
        "speaches-ai/piper-tr_TR-fahrettin-medium",
        "speaches-ai/piper-tr_TR-fettah-medium",
        "speaches-ai/piper-uk_UA-lada-x_low",
        "speaches-ai/piper-uk_UA-ukrainian_tts-medium",
        "speaches-ai/piper-vi_VN-25hours_single-low",
        "speaches-ai/piper-vi_VN-vais1000-medium",
        "speaches-ai/piper-vi_VN-vivos-x_low",
        "speaches-ai/piper-zh_CN-huayan-medium",
        "Stoned-Code/piper-en_US-glados-medium",
    ],
    audio_transcriptions_models=[
        "Systran/faster-whisper-large-v3",
        "deepdml/faster-whisper-large-v3-turbo-ct2",
        "mort666/faster-whisper-large-v2-th",
        "good-tape/faster-whisper-large-v3",
        "Zoont/faster-whisper-large-v3-turbo-int8-ct2",
        "KBLab/kb-whisper-base",
        "KBLab/kb-whisper-large",
        "KBLab/kb-whisper-small",
        "deepdml/faster-distil-whisper-large-v3.5",
        "Vinxscribe/biodatlab-whisper-th-medium-faster",
        "amir-eghtedar/whisper-large-fa-v1-fp16",
        "guillaumekln/faster-whisper-tiny",
        "guillaumekln/faster-whisper-tiny.en",
        "guillaumekln/faster-whisper-base.en",
        "guillaumekln/faster-whisper-base",
        "guillaumekln/faster-whisper-small.en",
        "guillaumekln/faster-whisper-small",
        "guillaumekln/faster-whisper-medium.en",
        "guillaumekln/faster-whisper-medium",
        "guillaumekln/faster-whisper-large-v2",
        "guillaumekln/faster-whisper-large-v1",
        "smcproject/vegam-whisper-medium-ml",
        "smcproject/vegam-whisper-medium-ml-fp16",
        "smcproject/vegam-whisper-medium-ml-int8",
        "smcproject/vegam-whisper-medium-ml-int8_float16",
        "bofenghuang/whisper-large-v2-cv11-french-ct2",
        "bofenghuang/whisper-large-v2-cv11-german-ct2",
        "arc-r/faster-whisper-large-v2-mix-jp",
        "arc-r/faster-whisper-large-v2-jp",
        "arc-r/faster-whisper-large-v2-Ko",
        "arc-r/faster-whisper-large-zh-cv11",
        "dwhoelz/whisper-large-pt-cv11-ct2",
        "dwhoelz/whisper-medium-pt-ct2",
        "jerichosiahaya/faster-whisper-medium-id",
        "YassineKader/faster-whisper-small-haitian",
        "arminhaberl/faster-whisper-tiny",
        "arminhaberl/faster-whisper-small",
        "arminhaberl/faster-whisper-medium",
        "arminhaberl/faster-whisper-large-v2",
        "arminhaberl/faster-whisper-large-v1",
        "arminhaberl/faster-whisper-base",
        "deepsync/whisper-large-v2-custom-hi",
        "davidggphy/whisper-small-dv-ct2",
        "mukowaty/faster-whisper-int8",
        "lorneluo/faster-whisper-large-v2",
        "lorneluo/whisper-small-ct2-int8",
        "Sagicc/faster-whisper-large-v2-sr",
        "bababababooey/faster-whisper-large-v3",
        "gradjitta/ct2-whisper-large-v3",
        "flyingleafe/faster-whisper-large-v3",
        "avans06/faster-whisper-large-v3",
        "Sagicc/faster-whisper-large-v3-sr",
        "BBLL3456/faster-whisper-large-V3",
        "trungkienbkhn/faster-whisper-large-v3",
        "Systran/faster-whisper-large-v2",
        "Systran/faster-whisper-large-v1",
        "Systran/faster-whisper-medium",
        "Systran/faster-whisper-medium.en",
        "Systran/faster-whisper-base",
        "Systran/faster-whisper-tiny",
        "Systran/faster-whisper-small",
        "Systran/faster-whisper-base.en",
        "Systran/faster-whisper-tiny.en",
        "Systran/faster-whisper-small.en",
        "lukas-jkl/faster-whisper-v3",
        "srd4/faster-whisper-large-v2",
        "Sagicc/faster-whisper-medium-sr",
        "srd4/faster-whisper-medium",
        "Systran/faster-distil-whisper-large-v2",
        "Systran/faster-distil-whisper-medium.en",
        "Systran/faster-distil-whisper-small.en",
        "JhonVanced/faster-whisper-large-v2",
        "JhonVanced/faster-whisper-large-v3",
        "JhonVanced/faster-whisper-large-v3-ja",
        "JhonVanced/whisper-large-v3-japanese-4k-steps-ct2",
        "firelily/quick-listing",
        "Sagicc/faster-whisper-large-sr-v2",
        "Necklace/faster-nb-whisper-large",
        "jvh/whisper-base-quant-ct2",
        "jvh/whisper-large-v2-quant-ct2",
        "jvh/whisper-medium-quant-ct2",
        "jvh/whisper-large-v3-quant-ct2",
        "distil-whisper/distil-large-v3-ct2",
        "Systran/faster-distil-whisper-large-v3",
        "Conner/docuvet-large-whisper",
        "mmalyska/distil-whisper-large-v3-pl-ct2",
        "srivatsavdamaraju/api",
        "aTrain-core/faster-distil-whisper-large-v2",
        "aTrain-core/faster-whisper-large-v3",
        "Einstellung/faster-distil-whisper-large-v3-es",
        "kotoba-tech/kotoba-whisper-v1.0-faster",
        "JhonVanced/sin2piusc-whisper-large-v2-10k-ct2",
        "JhonVanced/sin2piusc-whisper-large-v2-10k-ct2-int8_float32",
        "Ruben9999/whp",
        "ubunto/audio",
        "RaivisDejus/whisper-large-v3-lv-ailab-ct2",
        "jkawamoto/whisper-tiny-ct2",
        "kotoba-tech/kotoba-whisper-v2.0-faster",
        "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster",
        "deepdml/whisper-large-v3-turbo",
        "jootanehorror/faster-whisper-large-v3-turbo-ct2",
        "lukas-jkl/faster-whisper-large-v3-turbo",
        "jkawamoto/whisper-large-v3-ct2",
        "sasikr2/whisper-large-v3-turbo-ct2",
        "aTrain-core/faster-whisper-large-v3-turbo",
        "Infomaniak-AI/faster-whisper-large-v3-turbo",
        "imonkeycorp/whisper-large-v3-transform",
        "devilteo911/whisper-small-ita-ct2",
        "fatymatariq/faster-distil-whisper-large-v2",
        "beydogan/whisper-large-v3-turbo-german-ct2-16",
        "beydogan/whisper-large-v3-turbo-german-ct2",
        "adminbr/whisper-small-pt-ct2",
        "qbsmlabs/PhoWhisper-small",
        "qbsmlabs/PhoWhisper-medium",
        "qbsmlabs/PhoWhisper-large",
        "asadfgglie/faster-whisper-large-v3-zh-TW",
        "suzii/vi-whisper-large-v3-turbo-v1-ct2",
        "gongting/faster-whisper-large-v2",
        "collabora/faster-whisper-small-hindi",
        "ChrisTorng/whisper-large-v3-turbo-common_voice_19_0-zh-TW-ct2",
        "KBLab/kb-whisper-medium",
        "KBLab/kb-whisper-tiny",
        "gongting/faster-whisper-large-v3",
        "PierreMesure/kb-whisper-large-ct2",
        "PierreMesure/kb-whisper-medium-ct2",
        "PierreMesure/kb-whisper-small-ct2",
        "PierreMesure/kb-whisper-tiny-ct2",
        "mpasila/faster-whisper-Visual-novel-transcriptor",
        "jestillore/kb-whisper-small-ct2",
        "jestillore/kb-whisper-medium-ct2",
        "jestillore/kb-whisper-large-ct2",
        "fiifinketia/whisper-large-v3-turbo-akan",
        "distil-whisper/distil-large-v3.5-ct2",
        "sorendal/skyrim-whisper-small-int8",
        "sorendal/skyrim-whisper-base-int8",
        "Polopopi/Haru",
        "sorendal/skyrim-whisper-distil-small",
        "sorendal/skyrim-whisper-distil-small-int8",
        "sorendal/skyrim-whisper-tiny",
        "sorendal/skyrim-whisper-tiny-int8",
        "k1nto/Belle-whisper-large-v3-zh-punct-ct2",
        "Kelno/whisper-large-v3-french-distil-dec16-ct2",
        "erax-ai/EraX-WoW-Turbo-V1.1-CT2",
        "sorendal/skyrim-whisper-medium.en",
        "sorendal/skyrim-whisper-medium.en-int8",
        "Purfview/faster-distil-whisper-large-v3.5",
        "Luigi/whisper-small-zh_tw-ct2",
        "Vinxscribe/biodatlab-whisper-th-large-v3-faster",
        "Vinxscribe/kotoba-whisper-v2.2-faster",
        "theSuperShane/whisper-large-v3-ja",
        "john2223/medium-model",
        "pariya47/distill-whisper-th-large-v3-ct2",
        "collabora/faster-whisper-large-v2-hindi",
        "collabora/faster-whisper-medium-hindi",
        "jayant-yadav/ggml-kb-whisper-large_strict",
        "q-henric/kb-whisper-large",
        "jestillore/kb-whisper-large-strict-ct2",
        "Hexoplon/nb-whisper-large-distil-turbo-beta-ct2",
        "ragunath-ravi/quantized-whisper-mini-ta",
        "phate334/Breeze-ASR-25-ct2",
        "SoybeanMilk/faster-whisper-Breeze-ASR-25",
        "hasin023/faster-regional-ct2-cpu",
        "apigordyn/faster-whisper-large-v3-backup",
        "chris365312/whisper-large-v3-de-at-ct2",
        "Ash8181/whisper-large-v3-russian-ct2",
        "avazir/faster-distil-whisper-large-v3-ru",
        "avazir/faster-distil-whisper-large-v3-ru-int8",
        "syvai/faster-hviske-v3-conversation",
        "leophill/whisper-large-v3-turbo-sw-kinyarwanda-ct2",
        "leophill/whisper-large-v3-sn-kinyarwanda-ct2",
        "BabelfishAI/faster-whisper-cripser-ct2",
        "TopherAU/faster-whisper-distil-medium.en-int8",
        "TheTobyB/whisper-large-v3-turbo-german-ct2",
        "rtlingo/mobiuslabsgmbh-faster-whisper-large-v3-turbo",
        "RaivisDejus/whisper-large-v3-lv-ailab-ct2-int8",
        "TheChola/whisper-large-v3-turbo-german-faster-whisper",
        "nebi/whisper-large-v3-turbo-swiss-german-ct2",
        "nebi/whisper-large-v3-turbo-swiss-german-ct2-int8",
        "nekusu/faster-whisper-large-v3-turbo-latam-int8-ct2",
    ],
    models={},
)
for model in _const.audio_speech_models:
    _const.models[model] = "tts"
for model in _const.audio_transcriptions_models:
    _const.models[model] = "stt"


class ModelInstalledInfo(BaseModel):
    id: str
    type: str
    registered_name: str
    options: InstallModelIn


class InstalledInfo:
    def __init__(
        self,
        docker: DockerOptions,
        port: int,
        models: dict[str, ModelInstalledInfo],
        options: InstallServiceIn,
    ):
        self.docker = docker
        self.port = port
        self.models = models
        self.options = options


class SpeachesAIService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "speaches-ai"

    def _generate_config(self, info: InstalledInfo) -> ServiceConfig:
        return ServiceConfig(options=info.options, models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()])

    async def _install_core(self, options: InstallServiceIn) -> InstalledInfo:
        volumes = [f"{self._get_working_dir()}/cache:/home/ubuntu/.cache/huggingface/hub"]
        image = _const.image_gpu if options.gpu else _const.image_cpu

        docker_options = DockerOptions(
            name="speaches-ai",
            image=image,
            image_port=8000,
            use_gpu=options.gpu,
            volumes=volumes,
            env_vars={
                "ENABLE_UI": "False",
            },
            restart="unless-stopped",
            reset_uid=True,
        )
        port = await install_and_run_docker(self.application_context, docker_options)
        return InstalledInfo(docker=docker_options, port=port, models={}, options=options)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.values():
            if model.type == "tts":
                self.endpoint_registry.unregister_audio_speech(model.registered_name)
            if model.type == "stt":
                self.endpoint_registry.unregister_audio_transcriptions(model.registered_name)
        self.installed = None
        await uninstall_docker(self.application_context, info.docker)
        if options.purge:
            await self._clear_working_dir()

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, type in _const.models.items():
            installed = model_id in info.models
            if filters.installed is None or filters.installed == installed:
                out_list.append(RetrieveModelOut(id=model_id, service=self.get_id(), type=type, installed=installed))
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        type = _const.models[model_id]
        installed = model_id in info.models
        return RetrieveModelOut(id=model_id, service=self.get_id(), type=type, installed=installed)

    async def _install_model(self, model_id: str, options: InstallModelIn) -> None:
        info = self._check_installed()
        if model_id in info.models:
            return
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        type = _const.models[model_id]
        res = await fetch_from_localhost(info.port, f"/v1/models/{model_id}", "POST")
        if res.status_code != 200 and res.status_code != 201:
            print("Error when install model in speaches-ai", model_id, res.status_code, res.data)
            raise HTTPException(status_code=400, detail="Model not avaialble")
        registered_name = options.alias if options.alias is not None else model_id
        info.models[model_id] = ModelInstalledInfo(id=model_id, type=type, registered_name=registered_name, options=options)
        if type == "tts":
            self.endpoint_registry.register_audio_speech_as_proxy(
                registered_name, ProxyOptions(url=f"http://localhost:{info.port}/v1/audio/speech")
            )
        if type == "stt":
            self.endpoint_registry.register_audio_transcriptions_as_proxy(
                registered_name, ProxyOptions(url=f"http://localhost:{info.port}/v1/audio/transcriptions", form=True)
            )

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "tts":
            self.endpoint_registry.unregister_audio_speech(model.registered_name)
        if model.type == "stt":
            self.endpoint_registry.unregister_audio_transcriptions(model.registered_name)

        if options.purge:
            await fetch_from_localhost(info.port, f"/v1/models/{model_id}", "DELETE")
