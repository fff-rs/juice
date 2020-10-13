# Juice

This is the workspace project for 

 * [juice](https://github.com/spearow/juice/blob/master/juice/README.md) - machine learning frameworks for hackers
 * [coaster](https://github.com/spearow/juice/blob/master/coaster/README.md) - underlying math abstraction
 * [coaster-nn](https://github.com/spearow/juice/blob/master/coaster-nn/README.md)
 * [coaster-blas](https://github.com/spearow/juice/blob/master/coaster-blas/README.md)
 * [greenglas](https://github.com/spearow/juice/blob/master/greenglas/README.md) - a data preprocessing framework
 * [juice-examples](https://github.com/spearow/juice/blob/master/juice-examples/README.md) - mnist demo

 Please conduct the individual README.md files for more information.

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


### Dependencies

#### Cap'n'Proto

[cpanp is a data interchange format](https://capnproto.org/) that is used to store and load networks with weights for [Juice](https://github.com/spearow/juice/juice).

`capnproto` and `capnproto-libs` plus their development packages are the ones needed from your package manager.

#### Cuda

Getting the cuda libraries up poses to be the first road-block many users face.

To get things working one needs to set the following environment variables:

```zsh
# examplary paths, unlikely to work for your local setup!
export CUDNN_INCLUDE_DIR=/opt/cuda/include
export CUDNN_LIB_DIR=/opt/cuda/targets/x86_64-linux/lib/
export CUBLAS_INCLUDE_DIR=/opt/cuda/include
export CUBLAS_LIB_DIR=/opt/cuda/targets/x86_64-linux/lib/
```

depending on __your local__ installation setup.

The currently supported cuda version is `cuda-10` (details in #114 and #115 )

Note that you need a capable nvidia device in order to _run_ the cuda backend.

#### OpenCL

You need the apropriate loader and device libraries. Since the `OpenCL` backend is still WIP, this will be detailed at a later point of time.


#### BLAS

Blas is a linear algebra used by the `native` backend.

`openblas` or `blas` is required to be present. Choose explicitly via `BLAS_VARIANT`.

By default an attempt is made to resolve the library via `pkg-config`.

Overriding via 

```zsh
# examplary paths, unlikely to work for your local setup!
export BLAS_LIB_DIR=/opt/blas/lib64/
export BLAS_INCLUDE_DIR=/opt/blas/include/
```

is also supported.

Linkage for the blas library variant is determined by setting `BLAS_STATIC` to `1` or unsetting `BLAS_STATIC`.

