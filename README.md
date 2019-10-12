# Juice

This is the workspace projet for 

 * [juice](https://github.com/spearow/juice/blob/master/juice/README.md) - machine learning frameworks for hackers
 * [coaster](https://github.com/spearow/juice/blob/master/coaster/README.md) - underlying math abstraction
 * [coaster-nn](https://github.com/spearow/juice/blob/master/coaster-nn/README.md)
 * [coaster-blas](https://github.com/spearow/juice/blob/master/coaster-blas/README.md)
 * [greenglas](https://github.com/spearow/juice/blob/master/greenglas/README.md) - a data preprocessing framework
 * [juice-examples](https://github.com/spearow/juice/blob/master/juice-examples/README.md) - mnist demo

 Please conduct the individual README files for more information.

## [Juice](https://github.com/spearow/juice) Examples

CLI for running [juice](https://github.com/spearow/juice) examples. More examples and benchmark tests can be found at the [juice examples directory](https://github.com/spearow/juice#examples).

### Install CLI

**DISCLAIMER: Currently both CUDA and cuDNN are required for the examples to build.**

Compile and call the build.
```bash
# install rust, if you need to
curl -sSf https://static.rust-lang.org/rustup.sh | sh
# download the code
git clone git@github.com:spearow/juice.git && cd juice/juice-examples
# build the binary
cargo build --release
# and you should see the CLI help page
../target/release/juice-examples --help
# which means, you can run the examples from the juice-examples README
```
*Note for OSX El Capitan users: `openssl` no longer ships with OSX by default. `brew link --force openssl` should fix the problem. If not, [see this Github issue](https://github.com/sfackler/rust-openssl/issues/255) for more details.*
