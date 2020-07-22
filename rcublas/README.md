# rust-cuBLAS • [![Join the chat at https://gitter.im/spearow/coaster](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://ci.spearow.io/api/v1/teams/spearow/pipelines/juice/jobs/test-rust-cublas/badge)](https://ci.spearow.io/teams/spearow/pipelines/juice/jobs/test-rust-cublas) [![Crates.io](https://img.shields.io/crates/v/rcublas.svg)](https://crates.io/crates/rcublas) [![License](https://img.shields.io/crates/l/rcublas.svg)](LICENSE)

rust-cuBLAS provides a safe wrapper for [CUDA's cuBLAS][cublas] library, so you
can use cuBLAS comfortably and safely in your Rust application.

As cuBLAS currently relies on CUDA to allocate memory on the GPU, you might also
look into [rust-cuda][rust-cuda].



rust-cublas is part of the High-Performance Computation Framework [Coaster][coaster], for the
[BLAS Plugin][plugin]. For an easy, unified interface for BLAS operations, such as those provided by
cuBLAS, you might check out [Coaster][coaster].

For more information,

* see rust-cuBLAS's [Documentation](https://spearow.github.io/juice/rcublas/)
* or get in touch on [Gitter][chat]

[cublas]: https://developer.nvidia.com/cublas
[coaster]: https://github.com/spearow/juice/tree/master/coaster
[plugin]: https://github.com/spearow/juice/tree/master/coaster-blas
[spearow]: https://spearow.io/project/juice
[juice]: https://github.com/spearow/juice


## Getting Started

If you're using Cargo, just add rust-cuBLAS to your Cargo.toml:

    [dependencies]
    cublas = "0.2.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add cublas

[cargo-edit]: https://github.com/killercup/cargo-edit

## Building

The library can be built by entering `cublas/` and `cublas-sys/`, and issuing a
`cargo build` within each directory.

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
on the [Gitter Channel][chat].

[contributing]: CONTRIBUTING.md
[chat]: https://gitter.im/spearow/juice


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
