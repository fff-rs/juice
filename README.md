# Collenchyma • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma) [![Coverage Status](https://coveralls.io/repos/autumnai/collenchyma/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/collenchyma?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma)](https://crates.io/crates/collenchyma) [![License](https://img.shields.io/crates/l/collenchyma.svg)](LICENSE)

Collenchyma is an extensible, pluggable, backend-agnostic framework for parallel,
high-performance computations on CUDA, OpenCL and common host CPU. It is fast, easy
to build and provides an extensible Rust struct to execute operations on almost any
machine, even if it does not have CUDA or OpenCL capable devices.

Collenchyma's abstracts over the different computation languages (Native,
OpenCL, Cuda) and let's you run highly-performant code, thanks to easy
parallelization, on servers, desktops or mobiles without the need to adapt your
code for the machine you deploy to. Collenchyma does not require OpenCL or Cuda
on the machine and automatically falls back to the native host CPU, making your
application highly flexible and fast to build.

Collenchyma was started at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

* __Parallelizing Performance__<br/>
Collenchyma makes it easy to parallelize computations on your machine, putting
all the available cores of your CPUs/GPUs to use.
Collenchyma provides optimized operations through Plugins,
that you can use right away to speed up your application.

* __Easily Extensible__<br/>
Writing custom operations for GPU execution becomes easy with Collenchyma, as
it already takes care of Framework peculiarities, memory management, safety and other
overhead. Collenchyma provides Plugins (see examples below), that you can use to extend
the Collenchyma backend with your own, business-specific operations.

* __Butter-smooth Builds__<br/>
As Collenchyma does not require the installation of various frameworks and
libraries, it will not add significantly to the build time of your application.
Collenchyma checks at run-time if these frameworks can be used and gracefully
falls back to the standard, native host CPU if they are not.
No long and painful build procedures for you or your users.

For more information,

* see Collenchyma's [Documentation](http://autumnai.github.io/collenchyma)
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

> Disclaimer: Collenchyma is currently in a very early and heavy stage of
> development. If you are experiencing any bugs that are not due to not yet
> implemented features, feel free to create an issue.

[arrayfire]: https://github.com/arrayfire/arrayfire
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add Collenchyma to your Cargo.toml:

    [dependencies]
    collenchyma = "0.0.8"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma

[cargo-edit]: https://github.com/killercup/cargo-edit

## Plugins

You can easily extend Collenchyma's `Backend` with more backend-agnostic operations, through Plugins.
Plugins provide a set of related operations - BLAS would be a good example. To extend Collenchyma's `Backend`
with operations from a Plugin, just add a the desired Plugin crate to your Cargo.toml file.
Here is a list of available Collenchyma Plugins, that you can use right away for your own application, or
take as a starting point, if you would like to create your own Plugin.

* [BLAS][collenchyma-blas] - Collenchyma plugin for backend-agnostic Basic Linear Algebra Subprogram Operations.
* [NN][collenchyma-nn] - Collenchyma plugin for Neural Network related algorithms.

You can easily write your own backend-agnostic, parallel operations and provide it for others,
via a Plugin - we are happy to feature your Plugin here, just send us a PR.

[collenchyma-blas]: http://github.com/autumnai/collenchyma-blas
[collenchyma-nn]: http://github.com/autumnai/collenchyma-nn

## Examples

Collenchyma comes without any operations. The following examples therefore assumes,
that you have added both `collenchyma` and the Collenchyma Plugin `collenchyma-nn`
to your Cargo manifest.

```rust
extern crate collenchyma as co;
extern crate collenchyma_nn as nn;
use co::prelude::*;
use nn::*;

fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
    if let &mut MemoryType::Native(ref mut mem) = mem {
        let mut mem_buffer = mem.as_mut_slice::<T>();
        for (index, datum) in data.iter().enumerate() {
            mem_buffer[index] = *datum;
        }
    }
}

fn main() {
    // Initialize a CUDA Backend.
    let backend = Backend::<Cuda>::default().unwrap();
    // Initialize two SharedTensors.
    let mut x = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
    let mut result = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
    // Fill `x` with some data.
    let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
    let native = Backend::<Native>::default().unwrap();
    x.add_device(native.device()).unwrap(); // Add native host memory
    x.sync(native.device()).unwrap(); // Sync to native host memory
    write_to_memory(x.get_mut(native.device()).unwrap(), payload); // Write to native host memory.
    x.sync(backend.device()).unwrap(); // Sync the data to the CUDA device.
    // Run the sigmoid operation, provided by the NN Plugin, on your CUDA enabled GPU.
    backend.sigmoid(&mut x, &mut result).unwrap();
    // See the result.
    result.add_device(native.device()).unwrap(); // Add native host memory
    result.sync(native.device()).unwrap(); // Sync the result to host memory.
    println!("{:?}", result.get(native.device()).unwrap().as_native().unwrap().as_slice::<f32>());
}
```

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].
And high priority issues, that we could need your help with.

We have a mostly real-time collaboration culture and happens here on Github and
on the [Collenchyma Gitter Channel][gitter-collenchyma].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[issue-2]: https://github.com/autumnai/collenchyma/issues/2
[issue-4]: https://github.com/autumnai/collenchyma/issues/4
[issue-5]: https://github.com/autumnai/collenchyma/issues/5
[issue-6]: https://github.com/autumnai/collenchyma/issues/6
[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

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
