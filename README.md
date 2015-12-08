# collenchyma-NN • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma-nn.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma-nn) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma-nn)](https://crates.io/crates/collenchyma-nn) [![License](https://img.shields.io/crates/l/collenchyma-nn.svg)](LICENSE)

collenchyma-NN provides Neural Network related algorithms for [Collenchyma][collenchyma],
so you can use NN operations on servers, desktops or mobiles with OpenCL, CUDA
and common host CPU support.

If you would like to write your own backend-agnostic, high-performance library, you can  
* take this library as an example for basically copy&paste,
* glance over the docs for a broader overview
* and [notify us about your library][gitter-collenchyma] - we are happy to feature your Collenchyma library on the Collenchyma README.

collenchyma-NN was started at [Autumn][autumn] to support the Machine Intelligence
Framework [Leaf][leaf] with backend-agnostic, state-of-the-art performance.

For more information,

* see collenchyma-NN's [Documentation](http://autumnai.github.io/collenchyma-nn)
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

[collenchyma]: https://github.com/autumnai/collenchyma
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add collenchyma-NN to your Cargo.toml:

    [dependencies]
    collenchyma = "0.0.4"
    collenchyma-nn = "0.1.0"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma-nn

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

collenchyma-NN is released under the [MIT License][license].

[license]: LICENSE
