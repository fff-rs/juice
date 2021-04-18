#!/usr/bin/env sh
set -e
cargo check --tests
fly -t spearow login -n juice-crashtesters --concourse-url https://ci.spearow.io
fly -t spearow execute \
    --tag simsalabim \
    --tag framework:cuda \
    --tag framework:opencl \
    -c ./remote-test/test.yml \
    --input juice=.
