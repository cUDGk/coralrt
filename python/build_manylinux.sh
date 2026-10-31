#!/bin/bash
# The Linux wheel for PyPI, inside a manylinux_2_28 container (as .github/workflows/wheels.yml runs it):
#   docker run --rm -v "$PWD:/io" -w /io quay.io/pypa/manylinux_2_28_x86_64 bash python/build_manylinux.sh
# Rust, the crate's tests, build_wheel.py, then auditwheel: it checks that the library needs nothing newer than
# glibc 2.28 and tags the wheel manylinux_2_28 (PyPI refuses plain linux_x86_64). libusb is loaded at run time,
# so the wheel carries no other library.
set -eu
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"
cargo test --release
cd python
/opt/python/cp312-cp312/bin/python build_wheel.py
auditwheel repair --plat "$AUDITWHEEL_PLAT" -w wheelhouse dist/*.whl
