# collenchyma-BLAS • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma-blas.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma-blas) [![Coverage Status](https://coveralls.io/repos/autumnai/collenchyma-blas/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/collenchyma-blas?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma-blas)](https://crates.io/crates/collenchyma-blas) [![License](https://img.shields.io/crates/l/collenchyma-blas.svg)](LICENSE)

collenchyma-NN provides full BLAS support for [Collenchyma][collenchyma],
so you can use Basic Linear Algebra Subprograms on servers, desktops or mobiles,
GPUs, FPGAs or CPUS, without carrying about OpenCL or CUDA support on the
machine.

collenchyma-NN was started at [Autumn][autumn] for the Rust Machine Intelligence
Framework [Leaf][leaf].

For more information,

* see collenchyma-NN's [Documentation](http://autumnai.github.io/collenchyma-nn)
* visit [Collenchyma][collenchyma] for portable operations and other Plugins.
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

[collenchyma]: https://github.com/autumnai/collenchyma
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add collenchyma-BLAS to your Cargo.toml:

    [dependencies]
    collenchyma = "0.0.8"
    collenchyma-blas = "0.1.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma-blas

[cargo-edit]: https://github.com/killercup/cargo-edit

## Provided Operations

This Plugins provides the following operations to the Collenchyma Backend.
A `-` means not yet implemented.
More information can be found in the [Documentation][docs-ops].

| Operation            | CUDA       | OpenCL    | Native    |
|---                   |---         |---        |---        |
| **Full Level 1**     | cuBLAS     | -         | rblas     |
| Level 2              | -          | -         | -         |
| Level 3              |            |           |           |
| GEMM                 | cuBLAS     | -         | rblas     |


[docs-ops]: http://autumnai.github.io/collenchyma-blas/collenchyma_blas/plugin/trait.IBlas.html

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
