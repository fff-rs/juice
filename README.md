# [Leaf](https://github.com/autumnai/leaf) Examples

CLI for running leaf examples.

Compile and call the build.
```bash
cargo build --release
target/release/leaf-examples --help

```

## Datasets

The Datasets get not shipped with this repository, but you can load them via the
CLI. e.g. loading the MNIST Dataset

```bash
target/release/leaf-examples load-dataset mnist

# run the MNIST linear example
target/release/leaf-examples mnist linear --batch-size 10
# run the MNIST MLP (Multilayer Perceptron) example
target/release/leaf-examples mnist mlp --batch-size 5 --learning-rate 0.001
# run the MNIST Convolutional Neural Network example
target/release/leaf-examples mnist conv --batch-size 10 --learning-rate 0.002
```
