#!/bin/bash
# WSL: the op suite of the Windows side (target\ops: models and the results of tflite_runtime 2.5 for Windows, from
# tests\ops\build.py and official.py) run by the Linux build of coralrt. The differences left are the last digits
# of glibc's sin, cos, exp and the like (the Windows results come from the MSVC CRT).
set -eu
source "$(dirname "$0")/env.sh"
rm -rf "$HOME/ops"
cp -r "$REPO/target/ops" "$HOME/ops"
"$LIN/target/release/examples/op_check" "$HOME/ops" > "$HOME/ops.log" 2>&1 || true
grep -v "^match\|^RAN " "$HOME/ops.log" | tail -40
