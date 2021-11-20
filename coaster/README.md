# coaster • [![Join the chat at https://gitter.im/spearow/juice](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/spearow/juice?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://ci.spearow.io/api/v1/teams/spearow/pipelines/juice/jobs/test-coaster/badge)](https://ci.spearow.io/teams/spearow/pipelines/juice) [![Crates.io](https://img.shields.io/crates/v/coaster.svg)](https://crates.io/crates/coaster) [![dependency status](https://deps.rs/repo/github/spearow/coaster/status.svg)](https://deps.rs/repo/github/spearow/coaster) [![License](https://img.shields.io/crates/l/coaster.svg)](#license)

coaster is an extensible, pluggable, backend-agnostic framework for parallel,
high-performance computations on CUDA, OpenCL and common host CPU. It is fast, easy
to build and provides an extensible Rust struct to execute operations on almost any
machine, even if it does not have CUDA or OpenCL capable devices.

coaster's abstracts over the different computation languages (Native,
OpenCL, Cuda) and let's you run highly-performant code, thanks to easy
parallelization, on servers, desktops or mobiles without the need to adapt your
code for the machine you deploy to. coaster does not require OpenCL or Cuda
on the machine and automatically falls back to the native host CPU, making your
application highly flexible and fast to build.

coaster is powering [Juice][juice].

* __Parallelizing Performance__<br/>
coaster makes it easy to parallelize computations on your machine, putting
all the available cores of your CPUs/GPUs to use.
coaster provides optimized operations through Plugins,
that you can use right away to speed up your application.

* __Easily Extensible__<br/>
Writing custom operations for GPU execution becomes easy with coaster, as
it already takes care of Framework peculiarities, memory management, safety and other
overhead. coaster provides Plugins (see examples below), that you can use to extend
the coaster backend with your own, business-specific operations.

* __Butter-smooth Builds__<br/>
As coaster does not require the installation of various frameworks and
libraries, it will not add significantly to the build time of your application.
coaster checks at run-time if these frameworks can be used and gracefully
falls back to the standard, native host CPU if they are not.
No long and painful build procedures for you or your users.

For more information,

* see coaster's [Documentation][documentation]
* or get in touch via [Gitter][chat]

> Disclaimer: coaster is currently in a very early and heavy stage of
> development. If you are experiencing any bugs that are not due to not yet
> implemented features, feel free to create an issue.

[arrayfire]: https://github.com/arrayfire/arrayfire
[spearow]: https://spearow.io/project/juice
[juice]: https://github.com/spearow/juice
[spearow]: https://spearow.io/projects/coaster
[documentation]: https://spearow.github.io/coaster

## Getting Started

If you're using Cargo, just add coaster to your Cargo.toml:

```toml
    [dependencies]
    coaster = "0.2"
```

If you're using [Cargo Edit][cargo-edit], you can call:

```sh
    $ cargo add coaster
```

[cargo-edit]: https://github.com/killercup/cargo-edit

## Plugins

You can easily extend coaster's `Backend` with more backend-agnostic operations, through Plugins.
Plugins provide a set of related operations - BLAS would be a good example. To extend coaster's `Backend`
with operations from a Plugin, just add a the desired Plugin crate to your Cargo.toml file.
Here is a list of available coaster Plugins, that you can use right away for your own application, or
take as a starting point, if you would like to create your own Plugin.

* [BLAS][coaster-blas] - coaster plugin for backend-agnostic Basic Linear Algebra Subprogram Operations.
* [NN][coaster-nn] - coaster plugin for Neural Network related algorithms.

You can easily write your own backend-agnostic, parallel operations and provide it for others,
via a Plugin - we are happy to feature your Plugin here, just send us a PR.

[coaster-blas]: https://github.com/spearow/juice/tree/master/coaster-blas
[coaster-nn]: https://github.com/spearow/juice/tree/master/coaster-nn

## Examples

coaster comes without any operations. The following examples therefore assumes,
that you have added both `coaster` and the coaster Plugin `coaster-nn`
to your Cargo manifest.

```rust
use coaster as co;
use coaster_nn as nn;

use co::prelude::*;
use nn::*;

fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
	let mut mem_buffer = mem.as_mut_slice::<T>();
	for (index, datum) in data.iter().enumerate() {
	    mem_buffer[index] = *datum;
	}
}

fn main() {
    // Initialize a CUDA Backend.
    let backend = Backend::<Cuda>::default().unwrap();
    // Initialize two SharedTensors.
    let mut x = SharedTensor::<f32>::new(&(1, 1, 3));
    let mut result = SharedTensor::<f32>::new(&(1, 1, 3));
    // Fill `x` with some data.
    let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
    let native = Backend::<Native>::default().unwrap();
    write_to_memory(x.write_only(native.device()).unwrap(), payload); // Write to native host memory.
    // Run the sigmoid operation, provided by the NN Plugin, on your CUDA enabled GPU.
    backend.sigmoid(&mut x, &mut result).unwrap();
    // See the result.
    println!("{:?}", result.read(native.device()).unwrap().as_slice::<f32>());
}
```

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].
And high priority issues, that we could need your help with.

We have a mostly real-time collaboration culture and happens here on Github and
on the [Gitter Channel][chat].
You can also reach out to the Maintainer(s)
{[@drahnr][drahnr],}.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[contributing]: CONTRIBUTING.md
[chat]: https://gitter.im/spearow/juice
[drahnr]: https://github.com/drahnr

## Changelog

You can find the release history in the root file [CHANGELOG.md][changelog].

> A changelog is a log or record of all the changes made to a project, such as a website or software project, usually including such records as bug fixes, new features, etc. - [Wikipedia][changelog-quote]

We are using [Clog][clog], the Rust tool for auto-generating CHANGELOG files.

[changelog]: CHANGELOG.md
[changelog-quote]: https://en.wikipedia.org/wiki/Changelog
[Clog]: https://github.com/clog-tool/clog-cli

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
