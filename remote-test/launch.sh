#!/usr/bin/env sh
set -e
cargo check --tests
fly -t fff login -n juice-crashtesters --concourse-url https://ci.fff.rs
fly -t fff execute \
    --tag framework:cuda \
    --tag framework:opencl \
    -c ./test.yml \
    --input juice=..
