#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.
#
# Start or stop a throwaway Infra instance for the e2e suite, on the local Docker daemon.
#
#   tests/e2e/instance.sh start <image> <id>   # writes the DF_E2E_* settings to stdout (eval or append to a dotenv)
#   tests/e2e/instance.sh upgrade <image> <id> # replaces the Infra container with <image>, same storage, keys and URL
#   tests/e2e/instance.sh stop <id>            # removes everything the instance created, keeps pulled images
#   tests/e2e/instance.sh pull <image>         # docker pull, retried while CI may still be pushing the image
#   tests/e2e/instance.sh image <image>        # builds <image> from $E2E_BUILD_CONTEXT when set, otherwise pulls it
#
# <id> namespaces everything (container, network, service containers, storage), so several instances can share a
# daemon and `stop` removes only its own. CI passes the job id. Docker images are kept - that is the runner's cache.

set -euo pipefail

STORAGE_ROOT="${E2E_STORAGE_ROOT:-${TMPDIR:-/tmp}}"
# Infra logs; CI points this into the checkout (artifacts) and keeps the root-owned storage outside it.
LOG_DIR="${E2E_LOG_DIR:-$STORAGE_ROOT}"

names() {
  local id="$1"
  INFRA="df-e2e-$id"
  NETWORK="df-e2e-$id"
  SUFFIX="_e2e$id"
  STORAGE="$STORAGE_ROOT/df-e2e-$id"
}

# run_infra <image> <id> <host port or empty> <admin key> <server key>
run_infra() {
  local image="$1" id="$2" port="$3" admin_key="$4" server_key="$5"
  names "$id"

  # The daemon this script talks to, so Infra drives the same one: /run/user/<uid>/docker.sock on rootless Docker.
  local socket
  socket="$(docker_socket)"

  local docker_config=()
  if [ -f "$HOME/.docker/config.json" ]; then
    # Registry credentials for private service images (e.g. hub.simplito.com) - Infra pulls with its own client.
    docker_config=(-v "$HOME/.docker/config.json:/root/.docker/config.json:ro")
  fi

  # The storage dir is mounted at the same path: service containers bind-mount from it through the host daemon.
  # The `infra` alias matches a CLI install, where services such as doc-chunker reach Infra at http://infra:8086.
  docker run -d --name "$INFRA" --network "$NETWORK" --network-alias infra -p "127.0.0.1:$port:8086" \
    -v "$socket:/var/run/docker.sock" \
    -v "$STORAGE:$STORAGE" \
    "${docker_config[@]}" \
    -e DF_INFRA_ADMIN_API_KEY="$admin_key" \
    -e DF_INFRA_API_KEY="$server_key" \
    -e DF_NAME="$INFRA" \
    -e DF_STORAGE_DIR="$STORAGE" \
    -e DF_DOCKER_SUBNET="$NETWORK" \
    -e DF_COMPOSE_PREFIX="e2e${id}_" \
    -e DF_CONTAINER_NAME_SUFFIX="$SUFFIX" \
    -e DOCKER_HOST=unix:///var/run/docker.sock \
    "$image" >/dev/null
}

docker_socket() {
  local host
  host="$(docker context inspect --format '{{.Endpoints.docker.Host}}')"
  case "$host" in
    unix://*) echo "${host#unix://}" ;;
    *)
      echo "instance.sh needs a local Docker daemon on a unix socket, not $host" >&2
      return 1
      ;;
  esac
}

host_port() {
  docker port "$INFRA" 8086/tcp | head -n1 | sed 's/.*://'
}

# Infra appends the suffix to every service container it creates.
suffixed() {
  local names
  # A failing `docker ps` must fail the caller, not read as "no containers".
  names="$(docker ps "$@" --format '{{.Names}}')" || return
  grep -E -- "${SUFFIX}\$" <<<"$names" || true
}

container_env() {
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$INFRA" | sed -n "s/^$1=//p"
}

start() {
  local image="$1" id="$2"
  names "$id"
  local admin_key server_key
  admin_key="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  server_key="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"

  mkdir -p "$STORAGE"
  docker network create "$NETWORK" >/dev/null
  run_infra "$image" "$id" "" "$admin_key" "$server_key"

  echo "DF_E2E_BASE_URL=http://127.0.0.1:$(host_port)"
  echo "DF_E2E_ADMIN_API_KEY=$admin_key"
  echo "DF_E2E_API_KEY=$server_key"
}

upgrade() {
  local image="$1" id="$2"
  names "$id"
  local port admin_key server_key
  port="$(host_port)"
  admin_key="$(container_env DF_INFRA_ADMIN_API_KEY)"
  server_key="$(container_env DF_INFRA_API_KEY)"

  # A graceful stop, as in production: Infra stops its service containers itself (stop_containers_on_shutdown).
  docker stop -t 120 "$INFRA" >/dev/null
  docker logs "$INFRA" >"$LOG_DIR/df-e2e-$id-before-upgrade.log" 2>&1 || true
  docker rm "$INFRA" >/dev/null

  # Otherwise the new Infra would attach to still-running services and never start a stopped one.
  local running
  running="$(suffixed)"
  if [ -n "$running" ]; then
    echo "upgrade: Infra left service containers running after shutdown:" $running >&2
    exit 1
  fi
  run_infra "$image" "$id" "$port" "$admin_key" "$server_key"
}

# Every step runs even when an earlier one fails, so a stuck container never leaves tens of GB of storage behind;
# the exit status reports whether anything failed.
stop() {
  local id="$1" failed=0
  names "$id"
  local image
  image="$(docker inspect --format '{{.Config.Image}}' "$INFRA" 2>/dev/null || true)"
  docker logs "$INFRA" >"$LOG_DIR/df-e2e-$id.log" 2>&1 || true
  docker rm -f "$INFRA" >/dev/null 2>&1 || true

  local leftovers
  if leftovers="$(suffixed -a)"; then
    if [ -n "$leftovers" ]; then
      echo "$leftovers" | xargs docker rm -f >/dev/null || { echo "stop: could not remove $leftovers" >&2; failed=1; }
    fi
  else
    echo "stop: could not list containers" >&2
    failed=1
  fi
  docker network rm "$NETWORK" >/dev/null 2>&1 || true

  # Service containers write as root, so remove the storage from inside a container - the Infra image is already
  # local; busybox only when the Infra container never started.
  if [ -d "$STORAGE" ]; then
    docker run --rm -q --entrypoint rm -v "$STORAGE_ROOT:/storage" "${image:-busybox:1.37}" -rf "/storage/df-e2e-$id" \
      || { echo "stop: could not remove $STORAGE" >&2; failed=1; }
  fi
  return "$failed"
}

pull() {
  local image="$1" attempt
  for attempt in $(seq 1 60); do
    docker pull -q "$image" >/dev/null && return
    echo "$image not available yet (attempt $attempt/60), retrying in 20s..." >&2
    sleep 20
  done
  echo "$image could not be pulled" >&2
  exit 1
}

image() {
  local image="$1"
  if [ -n "${E2E_BUILD_CONTEXT:-}" ]; then
    docker build -q --build-arg DF_BUILD_COMMIT="${CI_COMMIT_SHA:-}" -t "$image" "$E2E_BUILD_CONTEXT" >/dev/null
  else
    pull "$image"
  fi
}

case "${1:-}" in
  start) start "${2:?image}" "${3:?id}" ;;
  upgrade) upgrade "${2:?image}" "${3:?id}" ;;
  stop) stop "${2:?id}" ;;
  pull) pull "${2:?image}" ;;
  image) image "${2:?image}" ;;
  *)
    echo "usage: $0 start <image> <id> | upgrade <image> <id> | stop <id> | pull <image> | image <image>" >&2
    exit 2
    ;;
esac
