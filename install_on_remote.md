# How to install and run deepfellow infra on remote machines
First we need to ensure we have installed docker, just, uv. 
install just from https://just.systems/man/en/introduction.html
install uv from https://docs.astral.sh/uv/getting-started/installation/ 
install docker from https://docs.docker.com/engine/install/


Then run the following commands 
```bash
git clone ssh://git@gitlab2.simplito.com:1022/df/deepfellow-infra.git
cd deepfellow-infra
uv sync
source .venv/bin/activate
just dev
```