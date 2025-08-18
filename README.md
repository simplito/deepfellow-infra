
# DeepFellow Infra

* **To część systemu DeepFellow, która udostępnia lokalną/prywatną infrastrukturę AI agentom DeepFellow**
  * LLMs, VDBs, GDBs, ... ATPs .. 

Ważne cechy:

* Ultra prosta instalacja.
* OpenAI-compatible model API - żeby móc używać endpointa DF Infra w róznych narzędziach 3rd party.
* Ultra prosta ścieżka skalowania, czyli odpowiedź na pytanie "Dokupiliśmy sobie skrzynkę z fajnym GPU - jak ją podczepić do istniejącej DF Infra?".
* Ogarnianie sytuacji skrzynek multi-GPU.
* Działanie na cloudowych nodach gpu.
* Pełna obsługa każdego infra-hosta komendą `deepfellow infra`.

## Założenia na pierwszy, szybki etap prac:

* Dodawane przez nas routes serwera Infra mają nie być async, żeby nie walczyć teraz z pythonowym async IO.
* Infra gada z klientami poprzez json/dict - w obie strony - i NIE budujemy specjalnych typów I/O tu w projekcie, a przynajmniej na pierwszym etapie rozwoju.
* Na dzień dobry nie bawimy się w streamowanie! Jak blokujące będą ślicznie działać, to streamowanie dorzucimy. 
* Nie robimy też na razie żadnej autoryzacji - to trzeba będzie ogarnąć całościowo, łącznie z DF Serverem.
* **Fokus na podpinaniu różnych narzędzi AI !** - na poczatku na szerokość, a nie na głebokość - to rozepnie też resztę systemu DF.
* ...rozmawiaj z MM.

## Install
You need python 3.13 with uv, to install dependencies:
```bash
uv sync
```

## Server start
You need [just](https://github.com/casey/just). To start server type:

```bash
just dev
```

## Turn on gemma3-1b service
```bash
curl -v -X POST http://localhost:7999/admin -H "Content-Type: application/json" -d '{"args": ["easyocr", "install"]}'
```

## Docker image

Build
```bash
docker build -t infra .
```

To run application in docker you need share `docker.sock`. For rootless it is `/run/user/$UID/docker.sock` for normal setup it is `/var/run/docker.sock`. To run docker image with rootless call:
```
docker run -it --rm \
  -p 8086:8086 \
  -v $PWD/storage:/app/storage \
  -v /run/user/$UID/docker.sock:/var/run/docker.sock \
  infra
```

If you want to increase workers count run like this:
```
docker run -it --rm \
  -p 8086:8086 \
  -v $PWD/storage:/app/storage \
  -v /run/user/$UID/docker.sock:/var/run/docker.sock \
  infra \
  bash -c "uv run uvicorn server.main:app --host 0.0.0.0 --port 8086 --workers 2"
```