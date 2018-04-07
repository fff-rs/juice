# coaster-BLAS • [![Join the chat at https://gitter.im/spearow/coaster-nn](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/spearow/coaster?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://ci.spearow.io/api/v1/pipelines/juice/jobs/test-coaster-blas/badge)](https://ci.spearow.io/teams/main/pipelines/leaf) [![Crates.io](https://img.shields.io/crates/v/coaster-blas.svg)](https://crates.io/crates/coaster-nn) [![License](https://img.shields.io/crates/l/coaster-blas.svg)](#license)

coaster-BLAS provides full BLAS support for [Coaster][coaster],
so you can use Basic Linear Algebra Subprograms on servers, desktops or mobiles,
GPUs, FPGAs or CPUS, without carrying about OpenCL or CUDA support on the
machine.

coaster-BLAS was started as collenchyma-BLAS at [Autumn][autumn] for the Rust Machine Intelligence
Framework [Leaf][leaf].

For more information,

* see coaster-BLAS's [Documentation](https://spearow.github.io/coaster-blas)
* visit [Coaster][coaster] for portable operations and other Plugins.
* or get in touch on [Gitter][gitter-coaster]

[coaster]: https://github.com/spearow/coaster
[autumn]: http://autumnai.com
[leaf]: https://github.com/spearow/leaf

## Getting Started

If you're using Cargo, just add coaster-BLAS to your Cargo.toml:

    [dependencies]
    coaster = "0.4"
    coaster-blas = "0.2.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add coaster-blas

[cargo-edit]: https://github.com/killercup/cargo-edit

## Provided Operations

This Plugins provides the following operations to the Coaster Backend.
A `-` means not yet implemented.
More information can be found in the [Documentation][docs-ops].

| Operation            | CUDA       | OpenCL    | Native    |
|---                   |---         |---        |---        |
| **Full Level 1**     | cuBLAS     | -         | rblas     |
| Level 2              | -          | -         | -         |
| Level 3              |            |           |           |
| GEMM                 | cuBLAS     | -         | rblas     |


[docs-ops]: https://spearow.github.io/coaster-blas/coaster_blas/plugin/trait.IBlas.html

Note that `blas` does not have all methods and thus fails to link, use `CARGO_BLAS=openblas` to compile for `native`

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].
And high priority issues, that we could need your help with.

We have a mostly real-time collaboration culture and happens here on Github and
on the [Coaster Gitter Channel][gitter-coaster].
You can also reach out to the Maintainers
{[@drahnr][drahnr]}.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as below, without any additional terms or
conditions.

[contributing]: CONTRIBUTING.md
[gitter-coaster]: https://gitter.im/spearow/coaster
[drahnr]: https://github.com/drahnr

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
