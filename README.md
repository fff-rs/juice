# Collenchyma • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma) [![Coverage Status](https://coveralls.io/repos/autumnai/collenchyma/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/collenchyma?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma)](https://crates.io/crates/collenchyma) [![License](https://img.shields.io/crates/l/collenchyma.svg)](LICENSE)

Collenchyma is an extensible, pluggable backend-agnostic framework for parallel computations
on CUDA, OpenCL and common host CPU. It is fast and easy to build and provides
a extensible Rust struct to run high-performance computation on almost any device,
even if it does not have CUDA or OpenCL capable devices.

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
Collenchyma also provides optimized operations, through Plugins, for popular libraries,
such as BLAS, that you can use right away to speed up your application.

* __Easily Extensible__<br/>
Writing custom operations for GPU execution becomes easy with Collenchyma, as
it already takes care of Framework peculiarities, memory management and other
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
> implemented features, feel free to create a issue.

[arrayfire]: https://github.com/arrayfire/arrayfire
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add Collenchyma to your Cargo.toml:

    [dependencies]
    collenchyma = "0.0.4"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma

[cargo-edit]: https://github.com/killercup/cargo-edit

## Plugins

You can extend the operations available for your the Collenchyma backend with Plugins.
Plugins are a common set of related operations such as BLAS. Just add a Collenchyma Plugin,
which is nothing but a Rust crate, with collenchyma to your Cargo.toml. Here are some
available Collenchyma Plugins.

* [BLAS][collenchyma-blas] - Collenchyma plugin for backend-agnostic Basic Linear Algebra Subprogram Operations.
* [NN][collenchyma-nn] - Collenchyma plugin for Neural Network related algorithms.

You can easily write your own backend-agnostic, parallel operations and provide it for others,
via a Plugin. We are happy to feature your Plugin here, just send us a PR.

[collenchyma-blas]: http://github.com/autumnai/collenchyma-blas
[collenchyma-nn]: http://github.com/autumnai/collenchyma-nn

## Examples

Backend with custom defined Framework and Device.

```
extern crate collenchyma as co;
use co::framework::IFramework;
use co::backend::{Backend, BackendConfig};
use co::frameworks::Native;
fn main() {
   let framework = Native::new(); // Initialize the Framework
   let hardwares = framework.hardwares(); // Now you can obtain a list of available hardware for that Framework.
   // Create the custom Backend by providing a Framework and one or many Hardwares.
   let backend_config = BackendConfig::new(framework, hardwares);
   let backend = Backend::new(backend_config);
   // You can now execute all the operations available, e.g.
   // backend.dot(x, y);
 }
```
Machine-agnostic Backend.

```
extern crate collenchyma as co;
use co::framework::IFramework;
use co::backend::{Backend, BackendConfig};
use co::frameworks::Native;
fn main() {
    // Not yet implemented.
    // No need to provide a Backend Configuration.
    let backend = Backend::new(None);
    // You can now execute all the operations available, e.g.
    // backend.dot(x, y);
}
```

## Benchmarks

The following benchmarks highlight the overhead of calling the underlying library implementations.

Operation                                    | Collenchyma (Native backend) | rust-blas
-------------------------------------------- | ---------------------------- | ----------
1000x Dot product of two vectors of size 100 | 48,870 ns (+/- 499) | 15,226 ns (+/- 244)
100x Dot product of two vectors of size 1000 | 9,997 ns (+/- 215) | 6,920 ns (+/- 179)
10x Dot product of two vectors of size 10000 | 10,958 ns (+/- 377) | 10,333 ns (+/- 460)
5x Dot product of two vectors of size 20000  | 10,784 ns (+/- 2,338) | 10,533 ns (+/- 1,981)

The overhead of Collenchyma becomes negligible when executing operations on vectors bigger than ~10000-20000 elements.  
Reducing this overhead is a big priority and [you can help!](https://github.com/autumnai/collenchyma/issues/13)

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].
And high priority issues, that we could need your help with.

* Finish the OpenCL implementation. [#2][issue-2]
* Finish the Cuda implementation. [#4][issue-4]
* Make the Backend machine-agnostic [#5][issue-5]
* Finish BLAS library for Native, OpenCL, Cuda [#6][issue-6]

We have a mostly real-time collaboration culture and happens here on Github and
on the [Collenchyma Gitter Channel][gitter-collenchyma].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

[issue-2]: https://github.com/autumnai/collenchyma/issues/2
[issue-4]: https://github.com/autumnai/collenchyma/issues/4
[issue-5]: https://github.com/autumnai/collenchyma/issues/5
[issue-6]: https://github.com/autumnai/collenchyma/issues/6
[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

## License

Collenchyma is released under the [MIT License][license].

[license]: LICENSE
