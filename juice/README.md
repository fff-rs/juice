# Juice • [![Join the chat at https://gitter.im/spearow/juice](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/spearow/juice?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://ci.spearow.io/api/v1/teams/spearow/pipelines/juice/jobs/test-juice/badge)](https://ci.spearow.io/teams/spearow/pipelines/juice) [![Crates.io](https://img.shields.io/crates/v/juice.svg)](https://crates.io/crates/juice) [![dependency status](https://deps.rs/repo/github/spearow/juice/status.svg)](https://deps.rs/repo/github/spearow/juice) [![License](https://img.shields.io/crates/l/juice.svg)](#license)
## Introduction

Juice is a open Machine Learning Framework for hackers to build classical, deep
or hybrid machine learning applications. It was inspired by the brilliant people
behind TensorFlow, Torch, Caffe, Rust and numerous research papers and brings
modularity, performance and portability to deep learning.

Juice has one of the simplest APIs, is lean and tries to introduce minimal
technical debt to your stack.

See the [Juice - Machine Learning for Hackers][juice-book] book for more.

> See more Deep Neural Networks benchmarks on [Deep Learning Benchmarks][deep-learning-benchmarks-website].

Juice is portable. Run it on CPUs, GPUs, and FPGAs, on machines with an OS, or on
machines without one. Run it with OpenCL or CUDA. Credit goes to
[Coaster][coaster] and Rust.

We see Juice as the core of constructing high-performance machine intelligence
applications. Juice's design makes it easy to publish independent modules to make
e.g. deep reinforcement learning, visualization and monitoring, network
distribution, [automated preprocessing][greenglas] or scaleable production
deployment easily accessible for everyone.

[caffe]: https://github.com/BVLC/caffe
[rust]: https://www.rust-lang.org/
[juice-book]: https://spearow.github.io/juice/book/juice.html
[tensorflow]: https://github.com/tensorflow/tensorflow
[benchmarks]: #benchmarks
[juice-examples]: #examples
[documentation]: http://spearow.github.io/juice

> Disclaimer: Juice is currently in an early stage of development.
> If you are experiencing any bugs with features that have been
> implemented, feel free to create a issue.

## Getting Started

### Documentation

To learn how to build classical, deep or hybrid machine learning applications with Juice, check out the [Juice - Machine Learning for Hackers][juice-book] book.

For additional information see the [Rust API Documentation][documentation].

Or start by running the **Juice examples**.

We are providing a set of [Juice examples][juice-examples], where we and
others publish executable machine learning models built with Juice. It features
a CLI for easy usage and has a detailed guide in the [project
README.md][juice-examples].

Juice comes with an examples directory as well, which features popular neural
networks (e.g. Alexnet, Overfeat, VGG). To run them on your machine, just follow
the install guide, clone this repoistory and then run

```bash
# The examples currently require CUDA support.
cargo run --release --no-default-features --features cuda --example benchmarks alexnet
```

[juice-examples]: https://github.com/spearow/juice-examples

### Installation

> Juice is build in [Rust][rust]. If you are new to Rust you can install Rust as detailed [here][rust_download].
We also recommend taking a look at the [official Rust - Getting Started Guide][rust_getting_started].

To start building a machine learning application (Rust only for now. Wrappers are welcome) and you are using Cargo, just add Juice to your `Cargo.toml`:

```toml
[dependencies]
juice = "0.2.3"
```

[rust_download]: https://www.rust-lang.org/downloads.html
[rust_getting_started]: https://doc.rust-lang.org/book/getting-started.html
[cargo-edit]: https://github.com/killercup/cargo-edit

If you are on a machine that doesn't have support for CUDA or OpenCL you
can selectively enable them like this in your `Cargo.toml`:

```toml
[dependencies]
juice = { version = "0.3", default-features = false }

[features]
default = ["native"] # include only the ones you want to use, in this case "native"
native  = ["juice/native"]
cuda    = ["juice/cuda"]
opencl  = ["juice/opencl"]
```

> More information on the use of feature flags in Juice can be found in [FEATURE-FLAGS.md](./FEATURE-FLAGS.md)

### Contributing

If you want to start hacking on Juice (e.g.[adding a new `Layer`](new-layer))
you should start with forking and cloning the repository.

We have more instructions to help you get started in the [CONTRIBUTING.md][contributing].

We also has a near real-time collaboration culture, which happens
here on Github and on the [Gitter Channel][chat].

> Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as below, without any additional terms or conditions.

[new-layer]: http://spearow.io/juice/book/create-new-layer.html
[contributing]: CONTRIBUTING.md
[chat]: https://gitter.im/spearow/juice
[drahnr]: https://github.com/drahnr

## Ecosystem / Extensions

Juice is designed to be as modular and extensible as possible. More helpful crates you can use with Juice:

- [**Greenglas**][greenglas]: Preprocessing Framework for Machine Learning
- [**Coaster**][coaster]: Portable, HPC-Framework on any hardware with CUDA, OpenCL, Rust

[greenglas]: https://github.com/spearow/juice/tree/master/greenglas
[coaster]: https://github.com/spearow/juice/tree/master/coaster

## Support / Contact

- With a bit of luck, you can find us online on the #rust-machine-learning IRC at irc.mozilla.org,
- but we are always approachable on [Gitter Channel][chat]
- For bugs and feature request, you can create a [Github issue][juice-issue]
- For more private matters, send us email straight to our inbox: hej@spearow.io

[juice-issue]: https://github.com/spearow/juice/issues

## Changelog

You can find the release history at the [CHANGELOG.md][changelog]. We are using [Clog][clog], the Rust tool for auto-generating CHANGELOG files.

[changelog]: CHANGELOG.md
[Clog]: https://github.com/clog-tool/clog-cli

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
