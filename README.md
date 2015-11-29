# Collenchyma • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma) [![Coverage Status](https://coveralls.io/repos/autumnai/collenchyma/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/collenchyma?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma)](https://crates.io/crates/collenchyma) [![License](https://img.shields.io/crates/l/collenchyma.svg)](LICENSE)

Collenchyma provides a common Rust interface to run operations on any Cuda or OpenCL supported device, making deployment of high-performance code as easy and platform-agnostic as common code.

Collenchyma's abstracts over the different computation languages (Native, OpenCL, Cuda) and let's you run highly-performant code, thanks to easy parallelization, on servers, desktops or mobiles without the need to adapt your code for the machine you deploy to. Collenchyma does not require OpenCL or Cuda on the machine and automatically falls back to the native host CPU, making your application highly flexible.

* __Parallelizing Performance__<br/>
Using the full potential of the hardware results in significantly better
performance for many computation intensive applications. With Collenchyma
you parallelize your computations across all the cores of your handware, instead
of just the one on your CPU. Collenchyma also provides optimized operations for
the most popular computations such as BLAS, that you can use right away.

* __Easily Extensible__<br/>
Making your application support custom operations and logic for GPU execution is
easy. No need to care about framework specifications, memory management or other
overhead. Extending the backend with your own operations is a straight-forward
process - define the kernel code and mount it on the backend.

* __Butter-smooth Builds__<br/>
Unlike [Arrayfire][arrayfire], Collenchyma does not add significantly to the
build time of your application, as it does not require the installation of
various frameworks and libraries. It will rather check at run-time if these
frameworks can be used and gracefully fall back to the standard, native host
CPU if they are not. No long and painful build procedures for you or your users.

Collenchyma was started at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

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
    collenchyma = "0.0.2"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma

[cargo-edit]: https://github.com/killercup/cargo-edit

## Example

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

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].

We have a mostly real-time collaboration culture and happens here on Github and
on the [Collenchyma Gitter Channels][gitter-collenchyma].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

## License

Collenchyma is released under the [MIT License][license].

[license]: LICENSE
