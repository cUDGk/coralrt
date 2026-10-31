#!/bin/bash
# WSL, after build.sh: the Linux wheel built with build_wheel.py, installed into a fresh venv (~/venv-test, which
# device.sh uses too), and tests/wheel_smoke.py run from outside the source tree
set -eu
source "$HOME/.cargo/env"
source "$(dirname "$0")/env.sh"
cd "$LIN/python"
if [ ! -x "$HOME/venv-build/bin/python" ]; then
  python3 -m venv "$HOME/venv-build"
  "$HOME/venv-build/bin/pip" install -q setuptools wheel
fi
rm -rf dist
"$HOME/venv-build/bin/python" build_wheel.py > "$HOME/wheel.log" 2>&1 || { tail -20 "$HOME/wheel.log"; exit 1; }
ls dist
rm -rf "$HOME/venv-test"
python3 -m venv "$HOME/venv-test"
"$HOME/venv-test/bin/pip" install -q dist/*.whl pillow
cd /tmp
"$HOME/venv-test/bin/python" -I "$LIN/tests/wheel_smoke.py" "$WORK/test_data"
