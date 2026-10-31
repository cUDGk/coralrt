#!/bin/bash
# WSL, after linux/build.sh: macOS build check of coralrt. There is no Apple SDK here, so it is cargo check: the
# Rust code for macOS compiles, and the Ooura FFT C file goes through clang for the Apple target with the few
# declarations it needs from shim/ (enough for an object file; nothing is linked).
set -u
source "$HOME/.cargo/env"
SHIM=$(cd "$(dirname "$0")/shim" && pwd)
cd "$HOME/coral/coralrt"
for t in x86_64-apple-darwin aarch64-apple-darwin; do
  rustup target add "$t" > /dev/null 2>&1
  echo "== $t"
  env SDKROOT=/ CC=clang CFLAGS="-nostdinc -I$SHIM" cargo check --release --lib --target "$t" > "$HOME/mac_$t.log" 2>&1
  echo "exit $?"
  grep -E "^(warning|error)" -A5 "$HOME/mac_$t.log" | head -30
  tail -2 "$HOME/mac_$t.log"
done
