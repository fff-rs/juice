# collenchyma-NN • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma-nn.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma-nn) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma-nn)](https://crates.io/crates/collenchyma-nn) [![License](https://img.shields.io/crates/l/collenchyma-nn.svg)](LICENSE)

collenchyma-NN provides Neural Network related algorithms for [Collenchyma][collenchyma].
Run NN operations on servers, desktops or mobiles, GPUs, FPGAs or CPUS, without
carrying about OpenCL or CUDA support on the machine.

collenchyma-NN was started at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

For more information,

* see collenchyma-NN's [Documentation](http://autumnai.github.io/collenchyma-nn)
* visit [Collenchyma][collenchyma] for more information about portable operations and other Plugins.
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

[collenchyma]: https://github.com/autumnai/collenchyma
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Provided Operations

This Plugins provides the following operations to the Collenchyma Backend.
Every Operation includes forward + backward. A `-` means not yet implemented.
More information can be found in the [Documentation][docs-ops].

| Operation            | CUDA       | OpenCL    | Native    |
|---	                 |---	        |---        |---        |
| Sigmoid  	           | cuDNN v3  	| -  	      | Rust	  	|
| SigmoidPointwise     | cuDNN v3  	| -  	      |   	      |
| ReLU  	             | cuDNN v3   | -  	      | Rust      |
| ReLUPointwise        | cuDNN v3   | -  	      |           |
| Tanh  	   	         | cudNN v3   | - 	      | Rust      |
| TanhPointwise  	   	 | cudNN v3   | - 	      |           |
|   	   	             |  	        |  	        |           |
| Normalization (LRN)  | cudNN v3   | - 	      | -         |
|   	   	             |  	        |  	        |           |
| Convolution          | cudNN v3   | - 	      | -         |
|   	   	             |  	        |  	        |           |
| Softmax              | cudNN v3   | - 	      | Rust      |
| LogSoftmax           | cudNN v3   | - 	      | Rust      |
|   	   	             |  	        |  	        |           |
| Pooling Max          | cudNN v3   | - 	      | -         |
| Pooling Avg          | cudNN v3   | - 	      | -         |

Kudos to [ehiggs][ehiggs], for implementing the native Rust operations.

[docs-ops]: http://autumnai.github.io/collenchyma-nn/collenchyma_nn/trait.NN.html
[ehiggs]: https://github.com/ehiggs

## Getting Started

If you're using Cargo, just add collenchyma-NN to your Cargo.toml:

    [dependencies]
    collenchyma = "0.0.8"
    collenchyma-nn = "0.3.4"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma-nn

[cargo-edit]: https://github.com/killercup/cargo-edit

## Usage

Bring the Plugin trait and the other important Collenchyma traits/structs in scope and
you will be able to execute the here provided operations on your Collenchyma Backend.

```rust
extern crate collenchyma as co;
extern crate collenchyma_nn as nn;
use co::prelude::*;
use nn::*;
fn main() {
    // Initialize a CUDA Backend.
    let backend = Backend::<Cuda>::default().unwrap();
    // Initialize two SharedTensors.
    // Usually you would want also fill them with data.
    // More infos about that in the Collenchyma README.md
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
on the [Collenchyma Gitter Channel][gitter-collenchyma].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

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
