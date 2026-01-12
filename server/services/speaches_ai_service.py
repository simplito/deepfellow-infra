# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speaches AI service."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel

from server.applicationcontext import get_base_url
from server.docker import DockerImage, DockerOptions
from server.endpointregistry import ProxyOptions, RegistrationId
from server.models.api import ModelProps
from server.models.models import (
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    ModelField,
    ModelInfo,
    ModelSpecification,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceProgress,
    ServiceField,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.services.base2_service import Base2Service, ModelConfig, ServiceConfig
from server.utils.core import (
    DownloadedPacket,
    PreDownloadPacket,
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    convert_size_to_bytes,
    try_parse_pydantic,
)
from server.utils.loading import Progress

ModelType = Literal["tts", "stt"]


class SpeachesModel(BaseModel):
    type: ModelType
    size: str


class SpeachesAiConst(BaseModel):
    image_gpu: DockerImage
    image_cpu: DockerImage
    audio_speech_models: list[tuple[str, str]]
    audio_transcriptions_models: list[tuple[str, str]]
    models: dict[str, SpeachesModel]


_const = SpeachesAiConst(
    image_gpu=DockerImage(name="ghcr.io/speaches-ai/speaches:0.9.0-rc.1-cuda", size="5.9 GB"),
    image_cpu=DockerImage(name="ghcr.io/speaches-ai/speaches:0.9.0-rc.1-cpu", size="1.6 GB"),
    audio_speech_models=[
        ("speaches-ai/Kokoro-82M-v1.0-ONNX-fp16", ""),
        ("speaches-ai/Kokoro-82M-v1.0-ONNX", ""),
        ("speaches-ai/Kokoro-82M-v1.0-ONNX-int8", ""),
        ("speaches-ai/piper-ar_JO-kareem-low", "61MB"),
        ("speaches-ai/piper-ar_JO-kareem-medium", ""),
        ("speaches-ai/piper-ca_ES-upc_ona-medium", ""),
        ("speaches-ai/piper-ca_ES-upc_ona-x_low", ""),
        ("speaches-ai/piper-ca_ES-upc_pau-x_low", ""),
        ("speaches-ai/piper-cs_CZ-jirka-low", ""),
        ("speaches-ai/piper-cs_CZ-jirka-medium", ""),
        ("speaches-ai/piper-cy_GB-gwryw_gogleddol-medium", ""),
        ("speaches-ai/piper-da_DK-talesyntese-medium", ""),
        ("speaches-ai/piper-de_DE-eva_k-x_low", "20MB"),
        ("speaches-ai/piper-de_DE-karlsson-low", ""),
        ("speaches-ai/piper-de_DE-kerstin-low", "61MB"),
        ("speaches-ai/piper-de_DE-mls-medium", ""),
        ("speaches-ai/piper-de_DE-pavoque-low", ""),
        ("speaches-ai/piper-de_DE-ramona-low", ""),
        ("speaches-ai/piper-de_DE-thorsten-high", "109MB"),
        ("speaches-ai/piper-de_DE-thorsten-low", ""),
        ("speaches-ai/piper-de_DE-thorsten-medium", "61MB"),
        ("speaches-ai/piper-de_DE-thorsten_emotional-medium", "74MB"),
        ("speaches-ai/piper-el_GR-rapunzelina-low", ""),
        ("speaches-ai/piper-en_GB-alan-low", ""),
        ("speaches-ai/piper-en_GB-alan-medium", ""),
        ("speaches-ai/piper-en_GB-alba-medium", "61MB"),
        ("speaches-ai/piper-en_GB-aru-medium", ""),
        ("speaches-ai/piper-en_GB-cori-high", ""),
        ("speaches-ai/piper-en_GB-cori-medium", ""),
        ("speaches-ai/piper-en_GB-jenny_dioco-medium", ""),
        ("speaches-ai/piper-en_GB-northern_english_male-medium", "61MB"),
        ("speaches-ai/piper-en_GB-semaine-medium", ""),
        ("speaches-ai/piper-en_GB-southern_english_female-low", ""),
        ("speaches-ai/piper-en_GB-vctk-medium", ""),
        ("speaches-ai/piper-en_US-amy-low", "61MB"),
        ("speaches-ai/piper-en_US-amy-medium", "61MB"),
        ("speaches-ai/piper-en_US-arctic-medium", ""),
        ("speaches-ai/piper-en_US-bryce-medium", "61MB"),
        ("speaches-ai/piper-en_US-danny-low", ""),
        ("speaches-ai/piper-en_US-hfc_female-medium", "61MB"),
        ("speaches-ai/piper-en_US-hfc_male-medium", ""),
        ("speaches-ai/piper-en_US-joe-medium", ""),
        ("speaches-ai/piper-en_US-john-medium", ""),
        ("speaches-ai/piper-en_US-kathleen-low", "61MB"),
        ("speaches-ai/piper-en_US-kristin-medium", "61MB"),
        ("speaches-ai/piper-en_US-kusal-medium", ""),
        ("speaches-ai/piper-en_US-l2arctic-medium", ""),
        ("speaches-ai/piper-en_US-lessac-high", "109MB"),
        ("speaches-ai/piper-en_US-lessac-low", ""),
        ("speaches-ai/piper-en_US-lessac-medium", "61MB"),
        ("speaches-ai/piper-en_US-libritts-high", "131MB"),
        ("speaches-ai/piper-en_US-libritts_r-medium", ""),
        ("speaches-ai/piper-en_US-ljspeech-high", ""),
        ("speaches-ai/piper-en_US-ljspeech-medium", ""),
        ("speaches-ai/piper-en_US-norman-medium", ""),
        ("speaches-ai/piper-en_US-ryan-high", "116MB"),
        ("speaches-ai/piper-en_US-ryan-low", ""),
        ("speaches-ai/piper-en_US-ryan-medium", "61MB"),
        ("speaches-ai/piper-es_ES-carlfm-x_low", ""),
        ("speaches-ai/piper-es_ES-davefx-medium", ""),
        ("speaches-ai/piper-es_ES-mls_10246-low", ""),
        ("speaches-ai/piper-es_ES-mls_9972-low", ""),
        ("speaches-ai/piper-es_ES-sharvard-medium", "74MB"),
        ("speaches-ai/piper-es_MX-ald-medium", ""),
        ("speaches-ai/piper-es_MX-claude-high", ""),
        ("speaches-ai/piper-fa_IR-amir-medium", ""),
        ("speaches-ai/piper-fa_IR-gyro-medium", ""),
        ("speaches-ai/piper-fi_FI-harri-low", ""),
        ("speaches-ai/piper-fi_FI-harri-medium", ""),
        ("speaches-ai/piper-fr_FR-gilles-low", ""),
        ("speaches-ai/piper-fr_FR-mls-medium", ""),
        ("speaches-ai/piper-fr_FR-mls_1840-low", ""),
        ("speaches-ai/piper-fr_FR-siwis-low", ""),
        ("speaches-ai/piper-fr_FR-siwis-medium", ""),
        ("speaches-ai/piper-fr_FR-tom-medium", ""),
        ("speaches-ai/piper-fr_FR-upmc-medium", "74MB"),
        ("speaches-ai/piper-hu_HU-anna-medium", ""),
        ("speaches-ai/piper-hu_HU-berta-medium", ""),
        ("speaches-ai/piper-hu_HU-imre-medium", ""),
        ("speaches-ai/piper-is_IS-bui-medium", ""),
        ("speaches-ai/piper-is_IS-salka-medium", ""),
        ("speaches-ai/piper-is_IS-steinn-medium", ""),
        ("speaches-ai/piper-is_IS-ugla-medium", ""),
        ("speaches-ai/piper-it_IT-paola-medium", "61MB"),
        ("speaches-ai/piper-it_IT-riccardo-x_low", ""),
        ("speaches-ai/piper-ka_GE-natia-medium", ""),
        ("speaches-ai/piper-kk_KZ-iseke-x_low", ""),
        ("speaches-ai/piper-kk_KZ-issai-high", ""),
        ("speaches-ai/piper-kk_KZ-raya-x_low", ""),
        ("speaches-ai/piper-lb_LU-marylux-medium", ""),
        ("speaches-ai/piper-lv_LV-aivars-medium", ""),
        ("speaches-ai/piper-ne_NP-google-medium", ""),
        ("speaches-ai/piper-ne_NP-google-x_low", ""),
        ("speaches-ai/piper-nl_BE-nathalie-medium", ""),
        ("speaches-ai/piper-nl_BE-nathalie-x_low", ""),
        ("speaches-ai/piper-nl_BE-rdh-medium", ""),
        ("speaches-ai/piper-nl_BE-rdh-x_low", ""),
        ("speaches-ai/piper-nl_NL-mls-medium", ""),
        ("speaches-ai/piper-nl_NL-mls_5809-low", ""),
        ("speaches-ai/piper-nl_NL-mls_7432-low", ""),
        ("speaches-ai/piper-no_NO-talesyntese-medium", ""),
        ("speaches-ai/piper-pl_PL-darkman-medium", "61MB"),
        ("speaches-ai/piper-pl_PL-gosia-medium", "61MB"),
        ("speaches-ai/piper-pl_PL-mc_speech-medium", ""),
        ("speaches-ai/piper-pl_PL-mls_6892-low", ""),
        ("speaches-ai/piper-pt_BR-edresson-low", ""),
        ("speaches-ai/piper-pt_BR-faber-medium", ""),
        ("speaches-ai/piper-zh_CN-huayan-x_low", ""),
        ("speaches-ai/piper-ro_RO-mihai-medium", ""),
        ("speaches-ai/piper-ru_RU-denis-medium", ""),
        ("speaches-ai/piper-ru_RU-dmitri-medium", ""),
        ("speaches-ai/piper-ru_RU-irina-medium", ""),
        ("speaches-ai/piper-ru_RU-ruslan-medium", ""),
        ("speaches-ai/piper-sk_SK-lili-medium", ""),
        ("speaches-ai/piper-sl_SI-artur-medium", ""),
        ("speaches-ai/piper-sr_RS-serbski_institut-medium", ""),
        ("speaches-ai/piper-sv_SE-nst-medium", "61MB"),
        ("speaches-ai/piper-sw_CD-lanfrica-medium", ""),
        ("speaches-ai/piper-tr_TR-dfki-medium", ""),
        ("speaches-ai/piper-tr_TR-fahrettin-medium", ""),
        ("speaches-ai/piper-tr_TR-fettah-medium", ""),
        ("speaches-ai/piper-uk_UA-lada-x_low", ""),
        ("speaches-ai/piper-uk_UA-ukrainian_tts-medium", ""),
        ("speaches-ai/piper-vi_VN-25hours_single-low", ""),
        ("speaches-ai/piper-vi_VN-vais1000-medium", "61MB"),
        ("speaches-ai/piper-vi_VN-vivos-x_low", ""),
        ("speaches-ai/piper-zh_CN-huayan-medium", "61MB"),
        ("Stoned-Code/piper-en_US-glados-medium", "61MB"),
    ],
    audio_transcriptions_models=[
        ("Systran/faster-whisper-large-v3", "2.9GB"),
        ("deepdml/faster-whisper-large-v3-turbo-ct2", "1.6GB"),
        ("mort666/faster-whisper-large-v2-th", ""),
        ("good-tape/faster-whisper-large-v3", ""),
        ("Zoont/faster-whisper-large-v3-turbo-int8-ct2", ""),
        ("KBLab/kb-whisper-base", ""),
        ("KBLab/kb-whisper-large", ""),
        ("KBLab/kb-whisper-small", ""),
        ("deepdml/faster-distil-whisper-large-v3.5", "1.5GB"),
        ("Vinxscribe/biodatlab-whisper-th-medium-faster", "1.5GB"),
        ("amir-eghtedar/whisper-large-fa-v1-fp16", ""),
        ("guillaumekln/faster-whisper-tiny", "75MB"),
        ("guillaumekln/faster-whisper-tiny.en", "75MB"),
        ("guillaumekln/faster-whisper-base.en", "141MB"),
        ("guillaumekln/faster-whisper-base", "142MB"),
        ("guillaumekln/faster-whisper-small.en", "464MB"),
        ("guillaumekln/faster-whisper-small", "464MB"),
        ("guillaumekln/faster-whisper-medium.en", "1.5GB"),
        ("guillaumekln/faster-whisper-medium", "1.5GB"),
        ("guillaumekln/faster-whisper-large-v2", "2.9GB"),
        ("guillaumekln/faster-whisper-large-v1", "2.9GB"),
        ("smcproject/vegam-whisper-medium-ml", ""),
        ("smcproject/vegam-whisper-medium-ml-fp16", ""),
        ("smcproject/vegam-whisper-medium-ml-int8", ""),
        ("smcproject/vegam-whisper-medium-ml-int8_float16", ""),
        ("bofenghuang/whisper-large-v2-cv11-french-ct2", ""),
        ("bofenghuang/whisper-large-v2-cv11-german-ct2", ""),
        ("arc-r/faster-whisper-large-v2-mix-jp", ""),
        ("arc-r/faster-whisper-large-v2-jp", ""),
        ("arc-r/faster-whisper-large-v2-Ko", ""),
        ("arc-r/faster-whisper-large-zh-cv11", ""),
        ("dwhoelz/whisper-large-pt-cv11-ct2", ""),
        ("dwhoelz/whisper-medium-pt-ct2", ""),
        ("jerichosiahaya/faster-whisper-medium-id", ""),
        ("YassineKader/faster-whisper-small-haitian", ""),
        ("arminhaberl/faster-whisper-tiny", "75MB"),
        ("arminhaberl/faster-whisper-small", ""),
        ("arminhaberl/faster-whisper-medium", ""),
        ("arminhaberl/faster-whisper-large-v2", ""),
        ("arminhaberl/faster-whisper-large-v1", ""),
        ("arminhaberl/faster-whisper-base", ""),
        ("deepsync/whisper-large-v2-custom-hi", ""),
        ("davidggphy/whisper-small-dv-ct2", ""),
        ("mukowaty/faster-whisper-int8", ""),
        ("lorneluo/faster-whisper-large-v2", ""),
        ("lorneluo/whisper-small-ct2-int8", ""),
        ("Sagicc/faster-whisper-large-v2-sr", ""),
        ("bababababooey/faster-whisper-large-v3", ""),
        ("gradjitta/ct2-whisper-large-v3", ""),
        ("flyingleafe/faster-whisper-large-v3", ""),
        ("avans06/faster-whisper-large-v3", ""),
        ("Sagicc/faster-whisper-large-v3-sr", ""),
        ("BBLL3456/faster-whisper-large-V3", ""),
        ("trungkienbkhn/faster-whisper-large-v3", ""),
        ("Systran/faster-whisper-large-v2", "2.9GB"),
        ("Systran/faster-whisper-large-v1", "2.9GB"),
        ("Systran/faster-whisper-medium", "1.5GB"),
        ("Systran/faster-whisper-medium.en", "1.5GB"),
        ("Systran/faster-whisper-base", "142MB"),
        ("Systran/faster-whisper-tiny", "75MB"),
        ("Systran/faster-whisper-small", "464MB"),
        ("Systran/faster-whisper-base.en", "141MB"),
        ("Systran/faster-whisper-tiny.en", "75MB"),
        ("Systran/faster-whisper-small.en", "464MB"),
        ("lukas-jkl/faster-whisper-v3", ""),
        ("srd4/faster-whisper-large-v2", ""),
        ("Sagicc/faster-whisper-medium-sr", ""),
        ("srd4/faster-whisper-medium", ""),
        ("Systran/faster-distil-whisper-large-v2", "1.5GB"),
        ("Systran/faster-distil-whisper-medium.en", "756MB"),
        ("Systran/faster-distil-whisper-small.en", "321MB"),
        ("JhonVanced/faster-whisper-large-v2", ""),
        ("JhonVanced/faster-whisper-large-v3", ""),
        ("JhonVanced/faster-whisper-large-v3-ja", ""),
        ("JhonVanced/whisper-large-v3-japanese-4k-steps-ct2", ""),
        ("firelily/quick-listing", ""),
        ("Sagicc/faster-whisper-large-sr-v2", ""),
        ("Necklace/faster-nb-whisper-large", ""),
        ("jvh/whisper-base-quant-ct2", ""),
        ("jvh/whisper-large-v2-quant-ct2", ""),
        ("jvh/whisper-medium-quant-ct2", ""),
        ("jvh/whisper-large-v3-quant-ct2", ""),
        ("distil-whisper/distil-large-v3-ct2", ""),
        ("Systran/faster-distil-whisper-large-v3", "1.5GB"),
        ("Conner/docuvet-large-whisper", ""),
        ("mmalyska/distil-whisper-large-v3-pl-ct2", ""),
        ("srivatsavdamaraju/api", ""),
        ("aTrain-core/faster-distil-whisper-large-v2", ""),
        ("aTrain-core/faster-whisper-large-v3", ""),
        ("Einstellung/faster-distil-whisper-large-v3-es", ""),
        ("kotoba-tech/kotoba-whisper-v1.0-faster", ""),
        ("JhonVanced/sin2piusc-whisper-large-v2-10k-ct2", ""),
        ("JhonVanced/sin2piusc-whisper-large-v2-10k-ct2-int8_float32", ""),
        ("Ruben9999/whp", ""),
        ("ubunto/audio", ""),
        ("RaivisDejus/whisper-large-v3-lv-ailab-ct2", ""),
        ("jkawamoto/whisper-tiny-ct2", ""),
        ("kotoba-tech/kotoba-whisper-v2.0-faster", ""),
        ("kotoba-tech/kotoba-whisper-bilingual-v1.0-faster", ""),
        ("deepdml/whisper-large-v3-turbo", ""),
        ("jootanehorror/faster-whisper-large-v3-turbo-ct2", ""),
        ("lukas-jkl/faster-whisper-large-v3-turbo", ""),
        ("jkawamoto/whisper-large-v3-ct2", ""),
        ("sasikr2/whisper-large-v3-turbo-ct2", ""),
        ("aTrain-core/faster-whisper-large-v3-turbo", ""),
        ("Infomaniak-AI/faster-whisper-large-v3-turbo", ""),
        ("imonkeycorp/whisper-large-v3-transform", ""),
        ("devilteo911/whisper-small-ita-ct2", ""),
        ("fatymatariq/faster-distil-whisper-large-v2", ""),
        ("beydogan/whisper-large-v3-turbo-german-ct2-16", ""),
        ("beydogan/whisper-large-v3-turbo-german-ct2", ""),
        ("adminbr/whisper-small-pt-ct2", ""),
        ("qbsmlabs/PhoWhisper-small", ""),
        ("qbsmlabs/PhoWhisper-medium", ""),
        ("qbsmlabs/PhoWhisper-large", ""),
        ("asadfgglie/faster-whisper-large-v3-zh-TW", ""),
        ("suzii/vi-whisper-large-v3-turbo-v1-ct2", ""),
        ("gongting/faster-whisper-large-v2", ""),
        ("collabora/faster-whisper-small-hindi", ""),
        ("ChrisTorng/whisper-large-v3-turbo-common_voice_19_0-zh-TW-ct2", ""),
        ("KBLab/kb-whisper-medium", ""),
        ("KBLab/kb-whisper-tiny", ""),
        ("gongting/faster-whisper-large-v3", ""),
        ("PierreMesure/kb-whisper-large-ct2", ""),
        ("PierreMesure/kb-whisper-medium-ct2", ""),
        ("PierreMesure/kb-whisper-small-ct2", ""),
        ("PierreMesure/kb-whisper-tiny-ct2", ""),
        ("mpasila/faster-whisper-Visual-novel-transcriptor", ""),
        ("jestillore/kb-whisper-small-ct2", ""),
        ("jestillore/kb-whisper-medium-ct2", ""),
        ("jestillore/kb-whisper-large-ct2", ""),
        ("fiifinketia/whisper-large-v3-turbo-akan", ""),
        ("distil-whisper/distil-large-v3.5-ct2", ""),
        ("sorendal/skyrim-whisper-small-int8", ""),
        ("sorendal/skyrim-whisper-base-int8", ""),
        ("Polopopi/Haru", ""),
        ("sorendal/skyrim-whisper-distil-small", ""),
        ("sorendal/skyrim-whisper-distil-small-int8", ""),
        ("sorendal/skyrim-whisper-tiny", ""),
        ("sorendal/skyrim-whisper-tiny-int8", ""),
        ("k1nto/Belle-whisper-large-v3-zh-punct-ct2", ""),
        ("Kelno/whisper-large-v3-french-distil-dec16-ct2", ""),
        ("erax-ai/EraX-WoW-Turbo-V1.1-CT2", ""),
        ("sorendal/skyrim-whisper-medium.en", ""),
        ("sorendal/skyrim-whisper-medium.en-int8", ""),
        ("Purfview/faster-distil-whisper-large-v3.5", ""),
        ("Luigi/whisper-small-zh_tw-ct2", ""),
        ("Vinxscribe/biodatlab-whisper-th-large-v3-faster", ""),
        ("Vinxscribe/kotoba-whisper-v2.2-faster", ""),
        ("theSuperShane/whisper-large-v3-ja", ""),
        ("john2223/medium-model", ""),
        ("pariya47/distill-whisper-th-large-v3-ct2", ""),
        ("collabora/faster-whisper-large-v2-hindi", ""),
        ("collabora/faster-whisper-medium-hindi", ""),
        ("jayant-yadav/ggml-kb-whisper-large_strict", ""),
        ("q-henric/kb-whisper-large", ""),
        ("jestillore/kb-whisper-large-strict-ct2", ""),
        ("Hexoplon/nb-whisper-large-distil-turbo-beta-ct2", ""),
        ("ragunath-ravi/quantized-whisper-mini-ta", ""),
        ("phate334/Breeze-ASR-25-ct2", ""),
        ("SoybeanMilk/faster-whisper-Breeze-ASR-25", ""),
        ("hasin023/faster-regional-ct2-cpu", ""),
        ("apigordyn/faster-whisper-large-v3-backup", ""),
        ("chris365312/whisper-large-v3-de-at-ct2", ""),
        ("Ash8181/whisper-large-v3-russian-ct2", ""),
        ("avazir/faster-distil-whisper-large-v3-ru", ""),
        ("avazir/faster-distil-whisper-large-v3-ru-int8", ""),
        ("syvai/faster-hviske-v3-conversation", ""),
        ("leophill/whisper-large-v3-turbo-sw-kinyarwanda-ct2", ""),
        ("leophill/whisper-large-v3-sn-kinyarwanda-ct2", ""),
        ("BabelfishAI/faster-whisper-cripser-ct2", ""),
        ("TopherAU/faster-whisper-distil-medium.en-int8", ""),
        ("TheTobyB/whisper-large-v3-turbo-german-ct2", ""),
        ("rtlingo/mobiuslabsgmbh-faster-whisper-large-v3-turbo", ""),
        ("RaivisDejus/whisper-large-v3-lv-ailab-ct2-int8", ""),
        ("TheChola/whisper-large-v3-turbo-german-faster-whisper", ""),
        ("nebi/whisper-large-v3-turbo-swiss-german-ct2", ""),
        ("nebi/whisper-large-v3-turbo-swiss-german-ct2-int8", ""),
        ("nekusu/faster-whisper-large-v3-turbo-latam-int8-ct2", ""),
    ],
    models={},
)
for model_tuple in _const.audio_speech_models:
    _const.models[model_tuple[0]] = SpeachesModel(type="tts", size=model_tuple[1])
for model_tuple in _const.audio_transcriptions_models:
    _const.models[model_tuple[0]] = SpeachesModel(type="stt", size=model_tuple[1])


@dataclass
class ModelInstalledInfo:
    id: str
    type: str
    registered_name: str
    options: InstallModelIn
    registration_id: RegistrationId
    model_path: Path

    def get_info(self) -> ModelInfo:
        """Get info."""
        return ModelInfo(spec=self.options.spec, registration_id=self.registration_id)


class SpeachesAIOptions(BaseModel):
    gpu: bool


class SpeachesAIModelOptions(BaseModel):
    alias: str | None = None


@dataclass
class InstalledInfo:
    docker: DockerOptions
    models: dict[str, ModelInstalledInfo]
    options: InstallServiceIn
    parsed_options: SpeachesAIOptions
    container_host: str
    container_port: int
    docker_exposed_port: int
    base_url: str


class SpeachesAIService(Base2Service[InstalledInfo]):
    def get_id(self) -> str:
        """Return the service id."""
        return "speaches-ai"

    def get_description(self) -> str:
        """Return the service description."""
        return "Self-hosted Speech-to-Text and Text-to-Speech model runner."

    def get_size(self) -> ServiceSize:
        """Return the service size."""
        return {"cpu": _const.image_cpu.size, "gpu": _const.image_gpu.size}

    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""
        return ServiceSpecification(
            fields=[
                ServiceField(type="bool", name="gpu", description="Run on GPU", required=False, default=self._has_gpu_for_spec()),
            ]
        )

    def get_model_spec(self) -> ModelSpecification:
        """Return the model specification."""
        return ModelSpecification(
            fields=[
                ModelField(type="text", name="alias", description="Model alias", required=False),
            ]
        )

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""
        return None

    def get_installed_info(self) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""
        return self._get_service_installed_info() if self.installed is None else self.installed.options.spec

    def _generate_config(self, info: InstalledInfo | None) -> ServiceConfig:
        return ServiceConfig(
            options=info.options if info else None,
            models=[ModelConfig(model_id=x.id, options=x.options) for x in info.models.values()] if info else [],
            custom=self.custom,
        )

    def _get_image(self, gpu: bool) -> DockerImage:
        return _const.image_gpu if gpu else _const.image_cpu

    async def _install_core(self, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfo, StreamChunk]:
        if "gpu" not in options.spec:
            options.spec["gpu"] = self.docker_service.has_gpu_support
        parsed_options = try_parse_pydantic(SpeachesAIOptions, options.spec)
        volumes = [f"{self._get_working_dir()}/cache:/home/ubuntu/.cache/huggingface/hub"]
        image = self._get_image(parsed_options.gpu)
        await self._verify_docker_image(image.name, options.ignore_warnings)

        async def func(stream: Stream[StreamChunk]) -> InstalledInfo:
            await self._docker_pull(image, stream)
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))
            subnet = self.docker_service.get_docker_subnet()
            docker_options = DockerOptions(
                name="speaches-ai",
                container_name=self.docker_service.get_docker_container_name("speaches-ai"),
                image=image.name,
                image_port=8000,
                use_gpu=parsed_options.gpu,
                volumes=volumes,
                env_vars={
                    "ENABLE_UI": "False",
                },
                restart="unless-stopped",
                user="0:0",
                subnet=subnet,
                healthcheck={
                    "test": "curl --fail http://localhost:8000/health || exit 1",
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": "3",
                    "start_period": "5s",
                },
            )
            docker_exposed_port = await self.docker_service.install_and_run_docker(docker_options)
            container_host = self.docker_service.get_container_host(subnet, docker_options.name)
            container_port = self.docker_service.get_container_port(subnet, docker_exposed_port, docker_options.image_port)
            info = InstalledInfo(
                docker=docker_options,
                models={},
                options=options,
                parsed_options=parsed_options,
                container_host=container_host,
                container_port=container_port,
                docker_exposed_port=docker_exposed_port,
                base_url=get_base_url(container_host, container_port),
            )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))
            return info

        return PromiseWithProgress(func=func)

    async def _uninstall(self, options: UninstallServiceIn) -> None:
        info = self._check_installed()
        for model in info.models.copy().values():
            if model.type == "tts":
                self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
            if model.type == "stt":
                self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)
        self.installed = None
        await self.docker_service.uninstall_docker(info.docker)
        if options.purge:
            await self._clear_working_dir()

    async def stop(self) -> None:
        """Stop the Speaches AI service Docker container."""
        info = self.installed
        if not info:
            return
        await self._stop_docker(info.docker)

    def get_docker_compose_file_path(self, model_id: str | None) -> Path:
        """Get docker compose file path."""
        info = self.installed
        if not info:
            raise HTTPException(400, "Service not installed")
        if model_id:
            raise HTTPException(400, "Docker is not bound with this object")
        return self.docker_service.get_docker_compose_file_path(info.docker.name)

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return True

    async def list_models(self, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""
        info = self._check_installed()
        out_list: list[RetrieveModelOut] = []
        for model_id, model in _const.models.items():
            installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
            if filters.installed is None or filters.installed == installed:
                out_list.append(
                    RetrieveModelOut(
                        id=model_id,
                        service=self.get_id(),
                        type=model.type,
                        installed=installed,
                        size=model.size,
                        spec=self.get_model_spec(),
                        has_docker=False,
                    )
                )
        return ListModelsOut(list=out_list)

    async def get_model(self, model_id: str) -> RetrieveModelOut:
        """Get the model."""
        info = self._check_installed()
        if model_id not in _const.models:
            raise HTTPException(status_code=400, detail="Model not found")
        model = _const.models[model_id]
        installed = info.models[model_id].get_info() if model_id in info.models else self._get_model_installed_info(model_id)
        return RetrieveModelOut(
            id=model_id,
            service=self.get_id(),
            type=model.type,
            installed=installed,
            size=model.size,
            spec=self.get_model_spec(),
            has_docker=False,
        )

    async def _install_model(self, model_id: str, options: InstallModelIn) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        parsed_model_options = try_parse_pydantic(SpeachesAIModelOptions, options.spec) if options.spec else SpeachesAIModelOptions()
        info = self._check_installed()
        if model_id in info.models:
            return PromiseWithProgress(value=InstallModelOut(status="OK", details="Already installed"))
        if model_id not in _const.models:
            raise HTTPException(400, "Model not found")
        model = _const.models[model_id]

        async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
            model_id_fixed = f"models--{model_id.replace('/', '--')}"

            models_dir = self._get_working_dir() / "cache"

            model_dir = models_dir / model_id_fixed
            model_dir.mkdir(parents=True, exist_ok=True)

            progress = Progress(convert_size_to_bytes(model.size) or 0)

            stream.emit(StreamChunkProgress(type="progress", stage="download", value=0))
            async for packet in self.model_downloader.hugging_face_repo_with_blobs_downloader.download(model_id, model_dir):
                if isinstance(packet, DownloadedPacket) and packet.downloaded_bytes_size != 0:
                    progress.add_to_actual_value(packet.downloaded_bytes_size)
                    stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress.get_percentage()))
                elif isinstance(packet, PreDownloadPacket):
                    if max := packet.file_bytes_size:
                        progress.set_max_value(max)

            stream.emit(StreamChunkProgress(type="progress", stage="download", value=1))

            stream.emit(StreamChunkProgress(type="progress", stage="install", value=0))

            registered_name = parsed_model_options.alias if parsed_model_options.alias else model_id
            info.models[model_id] = model_info = ModelInstalledInfo(
                id=model_id,
                type=model.type,
                registered_name=registered_name,
                options=options,
                registration_id="",
                model_path=model_dir,
            )
            if model.type == "tts":
                model_info.registration_id = self.endpoint_registry.register_audio_speech_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True),
                    options=ProxyOptions(url=f"{info.base_url}/v1/audio/speech", rewrite_model_to=model_id),
                    registration_options=None,
                )
            if model.type == "stt":
                model_info.registration_id = self.endpoint_registry.register_audio_transcriptions_as_proxy(
                    model=registered_name,
                    props=ModelProps(private=True),
                    options=ProxyOptions(url=f"{info.base_url}/v1/audio/transcriptions", rewrite_model_to=model_id),
                    registration_options=None,
                )
            stream.emit(StreamChunkProgress(type="progress", stage="install", value=1))
            return InstallModelOut(status="OK", details="Installed")

        return PromiseWithProgress(func=func)

    async def _uninstall_model(self, model_id: str, options: UninstallModelIn) -> None:
        info = self._check_installed()
        if model_id not in info.models:
            return
        model = info.models[model_id]
        del info.models[model_id]
        if model.type == "tts":
            self.endpoint_registry.unregister_audio_speech(model.registered_name, model.registration_id)
        if model.type == "stt":
            self.endpoint_registry.unregister_audio_transcriptions(model.registered_name, model.registration_id)

        if options.purge:
            shutil.rmtree(model.model_path)
