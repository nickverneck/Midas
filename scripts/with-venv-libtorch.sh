#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_PATH="$ROOT/.env"
if [[ -f "$ENV_PATH" ]]; then
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="${raw#"${raw%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "${line:0:1}" == "#" || "$line" != *"="* ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    if [[ "$val" == \"*\" && "$val" == *\" ]]; then
      val="${val:1:-1}"
    elif [[ "$val" == \'*\' && "$val" == *\' ]]; then
      val="${val:1:-1}"
    fi
    if [[ -n "$key" ]]; then
      export "$key=$val"
    fi
  done < "$ENV_PATH"
fi

VENV_DIR="$ROOT/.venv"
VENV_BIN="$VENV_DIR/bin"
VENV_PY="$VENV_BIN/python"
PYTHON="${PYTHON:-python3}"

if [[ -x "$VENV_PY" ]]; then
  export VIRTUAL_ENV="$VENV_DIR"
  export PATH="$VENV_BIN:${PATH:-}"
  PYTHON="$VENV_PY"
fi

if [[ -z "${LIBTORCH_USE_PYTORCH:-}" ]]; then
  export LIBTORCH_USE_PYTORCH=1
fi
if [[ -z "${LIBTORCH_BYPASS_VERSION_CHECK:-}" ]]; then
  export LIBTORCH_BYPASS_VERSION_CHECK=1
fi

torch_root=""
python_cmd=""
if [[ -n "$PYTHON" ]]; then
  if [[ -x "$PYTHON" ]]; then
    python_cmd="$PYTHON"
  elif command -v "$PYTHON" >/dev/null 2>&1; then
    python_cmd="$(command -v "$PYTHON")"
  fi
fi
if [[ -n "$python_cmd" ]]; then
  torch_root="$(
    "$python_cmd" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)' 2>/dev/null || true
  )"
fi

if [[ -n "$torch_root" ]]; then
  if [[ "${LIBTORCH_USE_PYTORCH}" == "1" ]]; then
    export LIBTORCH="$torch_root"
  else
    export LIBTORCH="${LIBTORCH:-$torch_root}"
  fi
fi

if [[ -z "${LIBTORCH:-}" || "${LIBTORCH_USE_PYTORCH}" == "1" ]]; then
  if [[ -d "$VENV_DIR/lib" ]]; then
    for cand in "$VENV_DIR"/lib/python*/site-packages/torch; do
      if [[ -d "$cand" ]]; then
        export LIBTORCH="$cand"
        break
      fi
    done
  fi
fi

if [[ -n "${LIBTORCH:-}" ]]; then
  torch_lib="$LIBTORCH/lib"
  if [[ -d "$torch_lib" ]]; then
    case "$(uname -s)" in
      Darwin)
        export DYLD_LIBRARY_PATH="${torch_lib}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
        export DYLD_FALLBACK_LIBRARY_PATH="${torch_lib}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
        ;;
      Linux)
        export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ;;
      *)
        export PATH="${torch_lib}${PATH:+:$PATH}"
        ;;
    esac
  fi
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $(basename "$0") command [args...]" >&2
  exit 1
fi

exec "$@"
