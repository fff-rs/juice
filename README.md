# rust-cuDNN • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/rust-cudnn.svg?branch=master)](https://travis-ci.org/autumnai/rust-cudnn) [![Coverage Status](https://coveralls.io/repos/autumnai/rust-cudnn/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/rust-cudnn?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/rust-cudnn)](https://crates.io/crates/rust-cudnn) [![License](https://img.shields.io/crates/l/rust-cudnn.svg)](LICENSE)

rust-cuDNN provides safe wrapper around [CUDA's cuDNN][cudnn] library, so you can use
it comfortably and safely in your Rust application.

As cuDNN relies on CUDA to allocate memory on the GPU and sync the data, you might also
check out [rust-cuda][rust-cuda].
To let your high-performance computations run on machines which might not be CUDA/cuDNN enabled,
you can check out [Collenchyma][collenchyma].

rust-cuDNN was started at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

For more information,

* see rust-cuDNN's [Documentation](http://autumnai.github.io/rust-cudnn)
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

## Example

Initialize a new cuDNN context and execute a cuDNN function.

```rust
extern crate cudnn;
use co::framework::IFramework;
use co::backend::{Backend, BackendConfig};
use co::frameworks::Native;
fn main() {
   let cudnn = Cudnn::new().unwrap(); // Initialize a new cuDNN context and allocates resources.
   let data_desc =
   let hardwares = framework.hardwares(); // Now you can obtain a list of available hardware for that Framework.
   // Create the custom Backend by providing a Framework and one or many Hardwares.
   let backend_config = BackendConfig::new(framework, hardwares);
   let backend = Backend::new(backend_config);
   // You can now execute all the operations available, e.g.
   // backend.dot(x, y);
 }
```

## Building

rust-cudnn depends on the cuDNN runtime libraries,
which can be obtained from [NVIDIA](https://developer.nvidia.com/cudnn).

### Manual Configuration

rust-cudnn's build script will by default attempt to locate `cudnn` via pkg-config. This will not work in some situations, for example, on systems that don't have pkg-config, when cross compiling, or when cuDNN is not installed in the default system library directory (e.g. `/usr/lib`).

The build script can be configured by exporting the following environment variables:

`CUDNN_LIB_DIR` - If specified, a directory that will be used to find cuDNN runtime libraries.  
`CUDNN_STATIC` - If specified, cuDNN libraries will be statically rather than dynamically linked.  
`CUDNN_LIBS` - If specified, will be used to find cuDNN libraries under a different name.  
If either `CUDNN_LIB_DIR` or `CUDNN_INCLUDE_DIR` are specified, then the build script will skip the pkg-config step.

If your also need to run the compiled binaries yourself, make sure that they are available:
```sh
# Linux; for other platforms consult the instructions that come with cuDNN
cd <cudnn_installpath>
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
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
