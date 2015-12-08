# collenchyma-BLAS • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma-blas.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma-blas) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma-blas)](https://crates.io/crates/collenchyma-blas) [![License](https://img.shields.io/crates/l/collenchyma-blas.svg)](LICENSE)

collenchyma-BLAS provides full BLAS support for [Collenchyma][collenchyma],
so you can use BLAS operations on servers, desktops or mobiles with OpenCL, CUDA
and common host CPU support.

If you would like to write your own backend-agnostic, high-performance library, you can  
* take this library as an example for basically copy&paste,
* glance over the docs for a broader overview
* and [notify us about your library][gitter-collenchyma] - we are happy to feature your Collenchyma plugin
on the Collenchyma README.

collenchyma-BLAS was started at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

For more information,

* see collenchyma-BLAS's [Documentation](http://autumnai.github.io/collenchyma-blas)
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

[collenchyma]: https://github.com/autumnai/collenchyma
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add collenchyma-BLAS to your Cargo.toml:

    [dependencies]
    collenchyma = "X"
    collenchyma-blas = "0.1.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma-blas

[cargo-edit]: https://github.com/killercup/cargo-edit

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

collenchyma-BLAS is released under the [MIT License][license].

[license]: LICENSE
