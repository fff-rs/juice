# rust-cuDNN • [![Join the chat at https://gitter.im/spearow/coaster](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://ci.spearow.io/api/v1/teams/spearow/pipelines/juice/jobs/test-rust-cudnn/badge)](https://ci.spearow.io/teams/spearow/pipelines/juice/jobs/test-rust-cudnn) [![Crates.io](https://img.shields.io/crates/v/rcudnn.svg)](https://crates.io/crates/rcudnn) [![License](https://img.shields.io/crates/l/rcudnn.svg)](LICENSE)
rust-cuDNN provides safe wrapper for [CUDA's cuDNN][cudnn] library, so you can use
it comfortably and safely in your Rust application.

As cuDNN relies on CUDA to allocate memory on the GPU, you might also look into [rust-cuda][rust-cuda].

rust-cudnn was developed at [Autumn][autumn] for the Rust Machine Intelligence Framework [Leaf][leaf].

rust-cudnn is part of the High-Performance Computation Framework [Collenchyma][collenchyma], for the
[Neural Network Plugin][nn]. For an easy, unified interface for NN operations, such as those provided by
cuDNN, you might check out [Collenchyma][collenchyma].

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
    cudnn = "1.5.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add cudnn

[cargo-edit]: https://github.com/killercup/cargo-edit

## Example

Using the high-level `Cudnn` interface.

```rust
extern crate rcudnn as cudnn;
extern crate libc;
use cudnn::{Cudnn, TensorDescriptor};
use cudnn::utils::{ScalParams, DataType};
fn main() {
    // Initialize a new cuDNN context and allocates resources.
    let cudnn = Cudnn::new().unwrap();
    // Create a cuDNN Tensor Descriptor for `src` and `dest` memory.
    let src_desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
    let dest_desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
    let acti = cudnn.init_activation().unwrap();
    // Obtain the `src` and memory pointer on the GPU.
    // NOTE: You wouldn't do it like that. You need to really allocate memory on the GPU with e.g. CUDA or Collenchyma.
    let src_data: *const ::libc::c_void = ::std::ptr::null();
    let dest_data: *mut ::libc::c_void = ::std::ptr::null_mut();
    // Now you can compute the forward sigmoid activation on your GPU.
    cudnn.sigmoid_forward::<f32>(&acti, &src_desc, src_data, &dest_desc, dest_data, ScalParams::default());
}
```

## Building

rust-cudnn depends on the cuDNN runtime libraries,
which can be obtained from [NVIDIA](https://developer.nvidia.com/cudnn).

### Manual Configuration

rust-cudnn's build script will by default attempt to locate `cudnn` via pkg-config.
This will not work in some situations, for example,
* on systems that don't have pkg-config,
* when cross compiling, or
* when cuDNN is not installed in the default system library directory (e.g. `/usr/lib`).

Therefore the build script can be configured by exporting the following environment variables:

* `CUDNN_LIB_DIR`<br/>
If specified, a directory that will be used to find cuDNN runtime libraries.
e.g. `/opt/cuda/lib`

* `CUDNN_STATIC`<br/>
If specified, cuDNN libraries will be statically rather than dynamically linked.

* `CUDNN_LIBS`<br/>
If specified, will be used to find cuDNN libraries under a different name.

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

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan


## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
