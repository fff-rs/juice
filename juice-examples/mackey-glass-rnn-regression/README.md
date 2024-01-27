# [Juice](https://github.com/spearow/juice) Examples

CLI for running [juice](https://github.com/spearow/juice) examples.

## Install CLI

```bash
# install rust, if you need to
curl -sSf https://static.rust-lang.org/rustup.sh | sh
# Download repository, and navigate to this example
git clone git@github.com:spearow/juice.git && cd juice/juice-examples/mackey-glass-rnn-regression
# build the binary
cargo build --release
```

## Environmental Variables

This example relies upon CUDA and CUDNN to build, and must be able to find both on your machine at build and runtime. The easiest way to ensure this
is to set the following environmental variables;

### RUSTFLAGS

Rustflags must be set to link natively to `cuda.lib` and `cudnn.h` in the pattern
```RUSTFLAGS=-L native={ CUDA LIB DIR} -L native={CUDNN HEADER DIRECTORY}```, or a single pattern of `-L native` if both files are located in the same directory.

### LLVM_CONFIG_PATH

`LLVM_CONFIG_PATH` must point to your llvm-config binary, including the binary itself, i.e.
`LLVM_CONFIG_PATH=D:\llvm\llvm-9.0.1.src\Release\bin\llvm-config.exe`

### CUDNN_INCLUDE_DIR

`CUDNN_INCLUDE_DIR` must point at the `\include` directory for your version of CUDA, i.e. for CUDA version 11.2 on windows it would be:

`CUDNN_INCLUDE_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`

## [Mackey-Glass](http://www.scholarpedia.org/article/Mackey-Glass_equation) Dataset

A generated version of Mackey-Glass is packaged with Juice, and packaged in a way suitable for RNN networks.

```bash
cd juice-examples/mackey-glass-rnn-regression
# Train a RNN Network (*nix)
../../target/release/example-rnn-regression train --learning-rate=0.01 --batch-size=40 SavedRNNNetwork.juice 
# Train a RNN Network (Windows)
..\..\target\release\example-rnn-regression.exe train --learning-rate=0.01 --batch-size=40 SavedRNNNetwork.juice 

# Test the RNN Network (*nix)
../../target/release/example-rnn-regression test --batch-size=40  SavedRNNNetwork.juice
# Test the RNN Network (Windows)
..\..\target\release\example-rnn-regression.exe test --batch-size=40 SavedRNNNetwork.juice
```
