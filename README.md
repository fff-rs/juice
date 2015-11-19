# Collenchyma • [![Join the chat at https://gitter.im/autumnai/collenchyma](https://img.shields.io/badge/gitter-join%20chat-brightgreen.svg)](https://gitter.im/autumnai/collenchyma?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/autumnai/collenchyma.svg?branch=master)](https://travis-ci.org/autumnai/collenchyma) [![Coverage Status](https://coveralls.io/repos/autumnai/collenchyma/badge.svg?branch=master&service=github)](https://coveralls.io/github/autumnai/collenchyma?branch=master) [![Crates.io](http://meritbadge.herokuapp.com/collenchyma)](https://crates.io/crates/collenchyma) [![License](https://img.shields.io/crates/l/collenchyma.svg)](LICENSE)

Collenchyma is a framework for fast, parallel and hardware-agnostic computation,
similar to [Arrayfire][arrayfire].

Collenchyma was started at [Autumn][autumn] to support fast and parallel
computations, at the Machine Intelligence Framework [Leaf][leaf], on various
backends such as OpenCL, CUDA, or native CPU.
Collenchyma is written in Rust, which allows for a modular and easily extensible
architecture and has no hard dependency on any drivers or libraries, which makes
it easy to use, as it removes long and painful build processes.

Collenchyma comes with a super simple API, which allows you to write code once
and then execute it on one or multiple devices (CPUs, GPUs) without the need to
care for the specific computation language (OpenCL, CUDA, native CPU) or the
underlying hardware.

For more information,

* see Collenchyma's [Documentation](http://autumnai.github.io/collenchyma)
* or get in touch on [Twitter][twitter-autumn] or [Gitter][gitter-collenchyma]

> Disclaimer: Collenchyma is currently in a very early and heavy stage of
> development. If you are experiencing any bugs that are not due to not yet
> implemented features, feel free to create a issue.

[arrayfire]: https://github.com/arrayfire/arrayfire
[autumn]: http://autumnai.com
[leaf]: https://github.com/autumnai/leaf
[twitter-autumn]: https://twitter.com/autumn_eng

## Getting Started

If you're using Cargo, just add Collenchyma to your Cargo.toml:

    [dependencies]
    collenchyma = "0.0.1"

If you're using [Cargo Edit][cargo-edit], you can call:

    $ cargo add collenchyma

[cargo-edit]: https://github.com/killercup/cargo-edit

## Contributing

Want to contribute? Awesome! We have
[instructions to help you get started contributing code or documentation][contributing].

We have a mostly real-time collaboration culture and happens here on Github and
on the [Collenchyma Gitter Channels][gitter-collenchyma].
You can also reach out to the Maintainers
{[@MJ][mj], [@hobofan][hobofan]}.

[contributing]: CONTRIBUTING.md
[gitter-collenchyma]: https://gitter.im/autumnai/collenchyma
[mj]: https://twitter.com/mjhirn
[hobofan]: https://twitter.com/hobofan

## License

Leaf is released under the [MIT License][license].

[license]: LICENSE
