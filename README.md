# coaster-NN • [![Join the chat at https://gitter.im/spearow/coaster-nn](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/spearow/coaster?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://ci.spearow.io/api/v1/teams/spearow/pipelines/juice/jobs/test-coaster-nn/badge)](https://ci.spearow.io/teams/spearow/pipelines/juice) [![Crates.io](https://img.shields.io/crates/v/coaster-nn.svg)](https://crates.io/crates/coaster-nn) [![License](https://img.shields.io/crates/l/coaster-nn.svg)](#license)

coaster-NN provides Neural Network related algorithms for [coaster][coaster].
Run NN operations on servers, desktops or mobiles, GPUs, FPGAs or CPUS, without
carrying about OpenCL or CUDA support on the machine.

coaster-NN was started as collenchyma-NN at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

For more information,

* see coaster-NN's [Documentation](http://spearow.github.io/coaster-nn)
* visit [coaster][coaster] for more information about portable operations and other Plugins.
* or get in touch on [Gitter][gitter-coaster]

[coaster]: https://github.com/spearow/coaster
[autumn]: http://autumnai.com
[leaf]: https://github.com/spearow/leaf

## Provided Operations

This Plugins provides the following operations to the coaster Backend.
Every Operation includes forward + backward. A `-` means not yet implemented.
More information can be found in the [Documentation][docs-ops].

| Operation            | CUDA              | OpenCL    | Native        |
|---                   |---                |---        |---            |
| Sigmoid              | cuDNN v5 or later | -         | Rust          |
| SigmoidPointwise     | cuDNN v5 or later | -         | Rust          |
| ReLU                 | cuDNN v5 or later | -         | Rust          |
| ReLUPointwise        | cuDNN v5 or later | -         | Rust          |
| Tanh                 | cuDNN v5 or later | -         | Rust          |
| TanhPointwise        | cuDNN v5 or later | -         | Rust          |
|                      |                   |           |               |
| Normalization (LRN)  | cuDNN v5 or later | -         | -             |
|                      |                   |           |               |
| Convolution          | cuDNN v5 or later | -         | Rust(forward) |
|                      |                   |           |               |
| Softmax              | cuDNN v5 or later | -         | Rust          |
| LogSoftmax           | cuDNN v5 or later | -         | Rust          |
|                      |                   |           |               |
| Pooling Max          | cuDNN v5 or later | -         | Rust(forward) |
| Pooling Avg          | cuDNN v5 or later | -         | -             |

Kudos to [ehiggs][ehiggs], for implementing the initial native Rust operations.

[docs-ops]: https://spearow.github.io/coaster-nn/coaster_nn/trait.NN.html
[ehiggs]: https://github.com/ehiggs

## Getting Started

If you're using Cargo, just add coaster-NN to your Cargo.toml:

    [dependencies]
    coaster = "0.1.0"
    coaster-nn = "0.4.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add coaster-nn

[cargo-edit]: https://github.com/killercup/cargo-edit

## Usage

Bring the Plugin trait and the other important coaster traits/structs in scope and
you will be able to execute the here provided operations on your coaster Backend.

```rust
extern crate coaster as co;
extern crate coaster_nn as nn;
use co::prelude::*;
use nn::*;
fn main() {
    // Initialize a CUDA Backend.
    let backend = Backend::<Cuda>::default().unwrap();
    // Initialize two SharedTensors.
    // Usually you would want also fill them with data.
    // More infos about that in the coaster README.md
    let mut x = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
    let mut result = SharedTensor::<f32>::new(backend.device(), &(1, 1, 3)).unwrap();
    // Use the operation provided by this Plugin.
    backend.sigmoid(&mut x, &mut result);
}
```

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].
And high priority issues, that we could need your help with.

We have a mostly real-time collaboration culture and happens here on Github and
on the [coaster Gitter Channel][gitter-coaster].
You can also reach out to the Maintainer(s)
{[drahnr][drahnr]}.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[contributing]: CONTRIBUTING.md
[gitter-coaster]: https://gitter.im/spearow/coaster
[drahnr]: https://github.com/drahnr

## Changelog

> *A changelog is a log or record of all the changes made to a project, such as a website or software project, usually including such records as bug fixes, new features, etc.* - [Wikipedia][changelog-quote]

You can find the release history at the [CHANGELOG][changelog] file.

We are using [Clog][clog], the Rust tool for auto generating CHANGELOG files.

[changelog]: CHANGELOG.md
[changelog-quote]: https://en.wikipedia.org/wiki/Changelog
[Clog]: https://github.com/clog-tool/clog-cli

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
