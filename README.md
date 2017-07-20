# [Juice](https://github.com/spearow/juice) Examples

CLI for running [juice](https://github.com/spearow/juice) examples. More examples and benchmark tests can be found at the [juice examples directory](https://github.com/spearow/juice#examples).

## Install CLI

**DISCLAIMER: Currently both CUDA and cuDNN are required for the examples to build.**

Compile and call the build.
```bash
# install rust, if you need to
curl -sSf https://static.rust-lang.org/rustup.sh | sh
# download the code
git clone git@github.com:spearow/juice-examples.git && cd juice-examples
# build the binary
cargo build --release
# and you should see the CLI help page
target/release/juice-examples --help
# which means, you can run the examples from below
```
*Note for OSX El Capitan users: `openssl` no longer ships with OSX by default. `brew link --force openssl` should fix the problem. If not, [see this Github issue](https://github.com/sfackler/rust-openssl/issues/255) for more details.*

## MNIST

The MNIST Datasets comes not shipped with this repository (it's too big), but you can load it directly via the
CLI.

```bash
# download the MNIST dataset.
target/release/juice-examples load-dataset mnist

# run the MNIST linear example
target/release/juice-examples mnist linear --batch-size 10
# run the MNIST MLP (Multilayer Perceptron) example
target/release/juice-examples mnist mlp --batch-size 5 --learning-rate 0.001
# run the MNIST Convolutional Neural Network example
target/release/juice-examples mnist conv --batch-size 10 --learning-rate 0.002
```
