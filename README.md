# [Leaf](https://github.com/autumnai/leaf) Examples

CLI for running leaf examples.

Compile and call the build.
```bash
$ cargo build
$ target/debug/leaf-examples --help
```

## Datasets

The Datasets get not shipped with this repository, but you can load them via the
CLI. e.g. loading the MNIST Dataset

```bash
$ target/debug/leaf-examples load-dataset mnist
```
