#!/bin/bash
# WSL: the committed state of the Windows repository (master) cloned into ~/coral, coralrt built with its examples
# (--lib too: the examples alone don't produce libcoralrt.so)
set -eu
source "$HOME/.cargo/env"
WIN=$(cd "$(dirname "$0")/../../.." && pwd)
if [ -d "$HOME/coral/.git" ]; then
  git -C "$HOME/coral" pull -q "$WIN" master
else
  git clone -q "$WIN" "$HOME/coral"
fi
cd "$HOME/coral/coralrt"
git log -1 --oneline
cargo build --release --lib --examples > "$HOME/build.log" 2>&1 || { tail -30 "$HOME/build.log"; exit 1; }
grep -E "^(warning|error)" "$HOME/build.log" || true
ls -la target/release/libcoralrt.so
