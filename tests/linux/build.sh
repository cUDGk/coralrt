#!/bin/bash
# WSL: the committed state of the Windows repository (its checked-out branch) cloned into ~/coral, coralrt built with
# its examples (--lib too: the examples alone don't produce libcoralrt.so)
set -eu
source "$HOME/.cargo/env"
source "$(dirname "$0")/env.sh"
if [ -d "$HOME/coral/.git" ]; then
  git -C "$HOME/coral" pull -q "$GIT" HEAD
else
  git clone -q "$GIT" "$HOME/coral"
fi
cd "$LIN"
git log -1 --oneline
cargo build --release --lib --examples > "$HOME/build.log" 2>&1 || { tail -30 "$HOME/build.log"; exit 1; }
grep -E "^(warning|error)" "$HOME/build.log" || true
ls -la target/release/libcoralrt.so
