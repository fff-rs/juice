# [Juice](https://github.com/spearow/juice) MNIST Classification Example

CLI for running [juice](https://github.com/spearow/juice) networks for the MNIST dataset.

## Installing CLI
Compile and call the build.
```bash
# install rust, if you need to
curl -sSf https://static.rust-lang.org/rustup.sh | sh
# download the code
git clone git@github.com:spearow/juice.git && cd juice-examples/mnist-image-multiclass-classification
# build the binary
cargo build --release
# and you should see the CLI help page
../target/release/juice-examples --help
# which means, you can run the examples from below
```
*Note for OSX El Capitan users: `openssl` no longer ships with OSX by default. `brew link --force openssl` should fix the problem. If not, [see this Github issue](https://github.com/sfackler/rust-openssl/issues/255) for more details.*

## Environmental Variables
Some of these examples rely upon CUDA and CUDNN to build, and Juice must be able to find both on your machine at build and runtime. The easiest way to ensure this 
is to set the following environmental variables; 

### RUSTFLAGS
Rustflags must be set to link natively to `cuda.lib` and `cudnn.h` in the pattern

```RUSTFLAGS=-L native={ CUDA LIB DIR } -L native={ CUDNN HEADER DIRECTORY }```
 
 or 

```RUSTFLAGS=-L native={ CUDA&CUDNN LIB Directory }```

if both files are located in the same directory.

### LLVM_CONFIG_PATH
`LLVM_CONFIG_PATH` must point to your llvm-config binary, including the binary itself, i.e.

`LLVM_CONFIG_PATH=D:\llvm\llvm-9.0.1.src\Release\bin\llvm-config.exe`

### CUDNN_INCLUDE_DIR
`CUDNN_INCLUDE_DIR` must point at the `\include` directory for your version of CUDA, i.e. for CUDA version 10.1 it would be:

`CUDNN_INCLUDE_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include`

## MNIST

The MNIST Datasets comes not shipped with this repository (it's too big), but you can load it directly via the
CLI.

```bash
# download the MNIST dataset.
../target/release/example-mnist-classification load-dataset mnist

# run the MNIST linear example
./target/release/example-mnist-classification mnist linear --batch-size 10
# run the MNIST MLP (Multilayer Perceptron) example
./target/release/example-mnist-classification mnist mlp --batch-size 5 --learning-rate 0.001
# run the MNIST Convolutional Neural Network example
./target/release/example-mnist-classification mnist conv --batch-size 10 --learning-rate 0.002
```

## Fashion-MNIST

The [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset is
also too big to be included, but it can be downloaded in the same way as MNIST:

```bash
# download the fashion-MNIST dataset.
../target/release/juice-examples load-dataset fashion

# run the fashion-MNIST linear example
../target/release/juice-examples fashion linear --batch-size 10
# run the fashion-MNIST MLP (Multilayer Perceptron) example
../target/release/juice-examples fashion mlp --batch-size 5 --learning-rate 0.001
# run the fashion-MNIST Convolutional Neural Network example
../target/release/juice-examples fashion conv --batch-size 10 --learning-rate 0.002
```
