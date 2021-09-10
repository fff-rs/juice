# Feature flags in Juice

## The problem(s)

Supporting different backends is an important concept in Juice.

Optimally we would like to always have to choice of running Juice on all backends.
However in reality there are some tradeoffs that have to be made.

One problem is that certain backends require the presence of special hardware to
run (CUDA needs NVIDIA GPUs and cuda drivers and libraries on the host, OpenCL requires a capable CPU/GPU and runtime libs), or the libraries to address them are not present on
the developers machine which is necessary for compilation.

Another challenge is that not all backends have support for the same operations,
which constrains neural networks with special requirements to the backends that
provide those operations.

Tracking issue for implementing a specialization based system is in [Issue #7](https://github.com/spearow/juice/issues/7).

## The solution

Feature flags are a well known concept to add opt-in functionality that is
not necessary for every use-case of a library and are a good solution to the first
problem.
Luckily, Cargo, Rust's package manager has built-in support for feature flags.

A simple dependency with additional features enabled in a `Cargo.toml` looks like this:
```toml
[dependencies]
juice = { version = "0.2", features = ["native", "cuda"] }
```

Feature flags are usually used in an additive way, and Juice is no exception.

Some individual backends might not implement all features, i.e. currently the `native` backend has an incomplete implementation of the `Convolution` Layer, as such it will panic on use. The `cuda` backend has a complete `Convolution` Layer, which **is available** since the CUDA backend implements the required traits.

Note that the `native` backend is always required for the time being, in order to load data to `opencl` and `cuda` frameworks.

The default features set is `cuda native`.

### In your project

The simple `Cargo.toml` example above works in simple cases but if you want
to provide the same flexibility of backends in your project, you can reexport
the feature flags.

A typical example (including coaster) would look like this:

```toml
[dependencies]
juice = { version = "0.3", default-features = false }
# the native coaster feature is neccesary to read/write tensors
coaster = { version = "0.2", default-features = false, features = ["native"] }

[features]
default = ["native"]
native  = ["juice/native"]
opencl  = ["juice/opencl", "coaster/opencl"]
cuda    = ["juice/cuda", "coaster/cuda"]

```

Building your project would then look like this:
```sh
# having both native and CUDA backends
# `native` is provided by default, and `cuda` explicitly specified by `--features cuda`
cargo build --features cuda
# unleashing CUDA
# `native` default not included because of `--no-default-features`, and `cuda` explicitly specified by `--features cuda`
cargo build --no-default-features --features cuda
```
