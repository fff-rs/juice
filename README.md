# rust-cuBLAS • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/rust-cublas.svg?branch=master)](https://travis-ci.org/autumnai/rust-cublas) [![Coverage Status](https://coveralls.io/repos/autumnai/rust-cublas/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/rust-cublas?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/cublas)](https://crates.io/crates/cublas) [![License](https://img.shields.io/crates/l/cublas.svg)](LICENSE)

rust-cuBLAS provides a safe wrapper for [CUDA's cuBLAS][cublas] library, so you
can use cuBLAS comfortably and safely in your Rust application.

As cuBLAS currently relies on CUDA to allocate memory on the GPU, you might also
look into [rust-cuda][rust-cuda].

rust-cublas was developed at [Autumn][autumn] for the Rust Machine Intelligence
Framework [Leaf][leaf].

rust-cublas is part of the High-Performance Computation Framework [Collenchyma][collenchyma], for the
[BLAS Plugin][plugin]. For an easy, unified interface for BLAS operations, such as those provided by
cuBLAS, you might check out [Collenchyma][collenchyma].

For more information,

* see rust-cuBLAS's [Documentation](http://autumnai.github.io/rust-cublas)
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

[cublas]: https://developer.nvidia.com/cublas
[rust-cuda]: https://github.com/autumnai/rust-cuda
[collenchyma]: https://github.com/autumnai/collenchyma
[plugin]: https://github.com/autumnai/collenchyma-blas
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add rust-cuBLAS to your Cargo.toml:

    [dependencies]
    cublas = "0.1.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add cublas

[cargo-edit]: https://github.com/killercup/cargo-edit

## Building

rust-cublas depends on the cuBLAS runtime libraries,
which can be obtained from [NVIDIA](https://developer.nvidia.com/cublas).

### Manual Configuration

rust-cublas's build script will by default attempt to locate `cublas` via pkg-config.
This will not work in some situations, for example,
* on systems that don't have pkg-config,
* when cross compiling, or
* when cuBLAS is not installed in the default system library directory (e.g. `/usr/lib`).

Therefore the build script can be configured by exporting the following environment variables:

* `CUBLAS_LIB_DIR`<br/>
If specified, a directory that will be used to find cuBLAS runtime libraries.
e.g. `/opt/cuda/lib`

* `CUBLAS_STATIC`<br/>
If specified, cuBLAS libraries will be statically rather than dynamically linked.

* `CUBLAS_LIBS`<br/>
If specified, will be used to find cuBLAS libraries under a different name.

If either `CUBLAS_LIB_DIR` or `CUBLAS_INCLUDE_DIR` are specified, then the build script will skip the pkg-config step.

If your also need to run the compiled binaries yourself, make sure that they are available:
```sh
# Linux; for other platforms consult the instructions that come with cuBLAS
cd <cublas_installpath>
export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
```

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].

We have a mostly real-time collaboration culture and happens here on Github and
on the [Collenchyma Gitter Channel][gitter-collenchyma].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

## Changelog

You can find the release history in the root file [CHANGELOG.md][changelog].

A changelog is a log or record of all the changes made to a project, such as a website or software project, usually including such records as bug fixes, new features, etc. - [Wikipedia][changelog-quote]

We are using [Clog][clog], the Rust tool for auto-generating CHANGELOG files.

[changelog]: CHANGELOG.md
[changelog-quote]: https://en.wikipedia.org/wiki/Changelog
[Clog]: https://github.com/clog-tool/clog-cli

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
