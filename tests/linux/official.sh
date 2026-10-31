#!/bin/bash
# WSL, after ops.sh: the official tflite_runtime 2.5 for Linux (Python 3.9, Coral's package repository) records the
# op suite's results on Linux into ~/ops_linux, then the Linux build of coralrt is compared with them. coralrt
# follows the Windows build of 2.5, which itself differs from the Linux build in about 150 cases.
set -eu
if [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
fi
UV="$HOME/.local/bin/uv"
if [ ! -x "$HOME/venv-rt/bin/python" ]; then
  "$UV" venv -q --python 3.9 "$HOME/venv-rt"
  "$UV" pip install -q --python "$HOME/venv-rt/bin/python" "numpy<2" tflite_runtime==2.5.0.post1 --extra-index-url https://google-coral.github.io/py-repo/
fi
"$HOME/venv-rt/bin/python" -c "import tflite_runtime, numpy; print('tflite_runtime', tflite_runtime.__version__, 'numpy', numpy.__version__)"
rm -rf "$HOME/ops_linux"
mkdir -p "$HOME/ops_linux"
cp -r "$HOME/ops/models" "$HOME/ops_linux/models"
"$HOME/venv-rt/bin/python" -I "$HOME/coral/coralrt/tests/ops/official.py" "$HOME/ops_linux" 2>&1 | tail -1
"$HOME/coral/coralrt/target/release/examples/op_check" "$HOME/ops_linux" > "$HOME/ops_linux.log" 2>&1 || true
grep -v "^match\|^RAN " "$HOME/ops_linux.log" | tail -40
