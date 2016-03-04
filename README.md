# [Leaf](https://github.com/autumnai/leaf) Examples

CLI for running leaf examples.

Compile and call the build.
```bash
$ cargo build
$ target/debug/leaf-examples --help
```
*Note for OSX El Capitan users: `openssl` no longer ships with OSX by default. `brew link --force openssl` should fix the problem. If not, [see this Github issue](https://github.com/sfackler/rust-openssl/issues/255) for more details.*

## Datasets

The Datasets get not shipped with this repository, but you can load them via the
CLI. e.g. loading the MNIST Dataset

```bash
$ target/debug/leaf-examples load-dataset mnist
```
