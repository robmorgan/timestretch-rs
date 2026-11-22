#!/bin/sh
# Build the browser version of the deck app into desktop/dist.
# Usage: build-web.sh [build|serve]   (needs trunk and the wasm32 target)
set -eu
cd "$(dirname "$0")"
# Trunk must run the wasm-bindgen CLI that matches the locked crate version.
TRUNK_TOOLS_WASM_BINDGEN="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print }' ../Cargo.lock)"
export TRUNK_TOOLS_WASM_BINDGEN
exec trunk "${1:-build}" --release
