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
*Note for OSX El Capitan users: `openssl` no longer ships with OSX by default. `brew link --force openssl` should fix the problem. If not, [see this Github issue](https://github.com/sfackler/rust-openssl/issues/255) for more details.*

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
`CUDNN_INCLUDE_DIR` must point at the `\include` directory for your version of CUDA, i.e. for CUDA version 10.1 it would be:

`CUDNN_INCLUDE_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include`

## [Mackey-Glass](http://www.scholarpedia.org/article/Mackey-Glass_equation) Dataset

A generated version of Mackey-Glass is packaged with Juice, and packaged in a way suitable for RNN networks. 

```bash
# Train a RNN Network (*nix)
./target/release/example-rnn-regression train --file=SavedRNNNetwork.juice --learningRate=0.01 --batchSize=40
# Train a RNN Network (Windows)
.\target\release\example-rnn-regression.exe train --file=SavedRNNNetwork.juice --learningRate=0.01 --batchSize=40

# Test the RNN Network (*nix)
../target/release/example-rnn-regression test --file=SavedRNNNetwork.juice
# Test the RNN Network (Windows)
cd ../target/release/ && example-rnn-regression.exe test --file=SavedRNNNetwork.juice
```
