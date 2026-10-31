#!/bin/bash
# WSL with the stick attached through usbipd-win: the 24 recorded pycoral cases (target\oracle, the outputs of the
# official pycoral on Windows) through the Linux build, from Rust and from Python (~/venv-test from wheel.sh).
#
# Getting the stick into WSL (usbipd-win 5.3, with the UsbDk filter of the official Edge TPU runtime installed):
#  1. Windows, elevated: tests\linux\usbipd_helper.ps1 force-binds the stick (UsbDk breaks a normal bind) and holds
#     until %TEMP%\coralrt_usbipd\done exists. A stick plugged in before usbipd-win was installed won't attach
#     even force-bound: replug it, then create %TEMP%\coralrt_usbipd\bind for the helper to bind it again.
#  2. Windows: usbipd attach --wsl --busid <busid> --auto-attach (no admin; the first tries often fail with
#     "Device in error state", the loop keeps trying).
#  3. A stick still in its bootloader (1a6e:089a): run this once. The firmware loads, but the reset that starts it
#     never reaches the stick through usbip, so it ends with "not found". usbipd detach --busid <busid> on Windows
#     restarts the stick, which then comes back with the firmware as 18d1:9302 and is attached again by the loop.
#  4. Run this. Afterwards stop the attach loop, usbipd detach, and create the done file to give the stick back.
set -u
source "$HOME/.cargo/env"
source "$(dirname "$0")/env.sh"
lsusb | grep -i -E "18d1:9302|1a6e:089a" || echo "no Coral in lsusb"
cd "$LIN"
echo "== Rust"
timeout 600 target/release/examples/oracle_check "$WORK/test_data" "$REPO/target/oracle" 2>&1 | tail -12
echo "== Python"
timeout 600 "$HOME/venv-test/bin/python" -I tests/oracle/check_python.py "$WORK/test_data" "$REPO/target/oracle" 2>&1 | tail -12
