#!/usr/bin/env bash

# set -euo pipefail

BUILD_DIR="${1:-build}"
BUILD_TYPE="${2:-Release}"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build "$BUILD_DIR" -j"$(nproc)"

echo "Build done. Binary: $BUILD_DIR/main"

cp "$BUILD_DIR/main" "$BUILD_DIR/../main"

./"$BUILD_DIR/../main"