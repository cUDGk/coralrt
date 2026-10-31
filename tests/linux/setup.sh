#!/bin/bash
# WSL Ubuntu as root: what building and running coralrt needs (C compiler for the Ooura FFT, libusb at run time,
# Python with venv for the bindings, clang for the macOS build check), and Rust through rustup.
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../coralrt/tests/linux/setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q build-essential clang curl git pkg-config libusb-1.0-0 usbutils python3 python3-venv python3-pip
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
"$HOME/.cargo/bin/rustc" --version
"$HOME/.cargo/bin/cargo" --version
python3 --version
