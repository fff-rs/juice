# rust-cuDNN • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/rust-cudnn?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/rust-cudnn.svg?branch=master)](https://travis-ci.org/autumnai/rust-cudnn) [![Coverage Status](https://coveralls.io/repos/autumnai/rust-cudnn/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/rust-cudnn?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/rust-cudnn)](https://crates.io/crates/rust-cudnn) [![License](https://img.shields.io/crates/l/rust-cudnn.svg)](LICENSE)

rust-cuDNN provides safe wrapper around [CUDA's cuDNN library][cudnn], so you can use
it comfortably and safely in your Rust application.

If you want high-performance computation in Rust and want to deploy it to machines
which are not CUDA/cuDNN enabled, you can check out [Collenchyma][collenchyma].
For more general CUDA computations you can check out [rust-cuda][rust-cuda].

rust-cuDNN was started at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

For more information,

* see rust-cuDNN's [Documentation](http://autumnai.github.io/collenchyma)
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

[cudnn]: https://developer.nvidia.com/cudnn
[rust-cuda]: https://github.com/autumnai/rust-cuda
[collenchyma]: https://github.com/autumnai/collenchyma
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add rust-cuDNN to your Cargo.toml:

    [dependencies]
    cudnn = "0.1.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add cudnn

[cargo-edit]: https://github.com/killercup/cargo-edit

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

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].
And high priority issues, that we could need your help with.

We have a mostly real-time collaboration culture and happens here on Github and
on the [Collenchyma Gitter Channel][gitter-collenchyma].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

## License

rust-cuDNN is released under the [MIT License][license].

[license]: LICENSE
