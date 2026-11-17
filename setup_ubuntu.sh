#!/usr/bin/env bash
# One-step setup and quick run on Ubuntu 22.04 or later (also WSL).
#
#   bash setup_ubuntu.sh            # install everything, then ./run_all.sh quick my_results
#   bash setup_ubuntu.sh --no-run   # install only
#
# Installs the system packages, finds Python 3.12 or 3.13 (or installs 3.13
# with uv if the system has another version), creates .venv, installs
# requirements.txt (Pyfhel compiles from source: 5 to 10 minutes) and runs the
# quick reproduction into my_results/ so the committed results/ stay untouched.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${SKIP_APT:-0}" != "1" ]]; then
  echo "== Installing system packages (you may be asked for your password)"
  sudo apt-get update
  sudo apt-get install -y git curl build-essential python3-dev python3-venv
fi

supported() {
  command -v "$1" >/dev/null 2>&1 &&
    "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) else 1)' 2>/dev/null
}

PY=""
for candidate in python3.13 python3.12 python3; do
  if supported "$candidate"; then PY="$candidate"; break; fi
done

if [[ -z "$PY" ]]; then
  echo "== System Python is $(python3 --version 2>&1); installing Python 3.13 with uv"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv python install 3.13
  PY="$(uv python find 3.13)"
fi
echo "== Using $("$PY" --version) at $(command -v "$PY" || echo "$PY")"

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
echo "== Installing Python packages (Pyfhel compiles from source: 5 to 10 minutes)"
python -m pip install -r requirements.txt

if [[ "${1:-}" == "--no-run" ]]; then
  echo "Setup done. Activate with: source .venv/bin/activate"
  exit 0
fi

echo "== Running the quick reproduction"
./run_all.sh quick my_results
echo
echo "All done. Results are in $(pwd)/my_results/"
echo "To use the environment again later: cd $(pwd) && source .venv/bin/activate"
