#!/usr/bin/env bash

# set -euo pipefail

mkdir build/
cd build/
cmake ..
make

cp ./main ../main
cd ..
./main