# Juice - Machine Learning for Hackers

> Our life is frittered away by detail. Simplify, simplify. -
> _Henry David Thoreau_

This short book teaches you how you can build machine learning applications using
[Juice][juice].

Juice is a Machine Intelligence Framework engineered by hackers, not scientists.
It has a very simple API consisting of [Layers][layers] and [Solvers][solvers], with which
you can build classical machine as well as deep learning and other fancy machine
intelligence applications.

Juice was inspired by the brilliant people behind TensorFlow, Torch, Caffe,
[Rust][rust] and numerous research papers and brings modularity, performance and
portability to deep learning.

<br/>
<br/>

> To make the most of the book, a basic understanding of the fundamental concepts
> of machine and deep learning is recommended. Good resources to get you from
> zero to almost-ready-to-build-machine-learning-applications:
>
> * [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) or
> * [Stanford Course on (Convolutional) Neural Networks](http://cs231n.github.io/)
>
> And if you already have some experience, [A 'brief' history of Deep Learning](http://www.andreykurenkov.com/writing/a-brief-history-of-neural-nets-and-deep-learning/) or [The Glossary](./deep-learning-glossary.html)
> might prove informative.

Both machine and deep learning are really easy with Juice.

Construct a [Network](./deep-learning-glossary.html#Network) by chaining [Layers](./deep-learning-glossary.html#Layer).
Then optimize the network by feeding it examples.
This is why Juice's entire API consists of only two concepts: [Layers][layers]
and [Solvers][solvers]. Use layers to construct almost any kind of model: deep,
classical, stochastic or hybrids, and solvers for executing and optimizing the
model.

This is already the entire API for machine learning with Juice. To learn how
this is possible and how to build machine learning applications, refer to chapters
[2. Layers](./layers.html) and [3. Solvers](./solvers.html). Enjoy!

[juice]: https://github.com/spearow/juice
[layers]: ./layers.html
[solvers]: ./solvers.html
[rust]: https://rust-lang.org
[github]: https://github.com/spearow/juice
[gitter]: https://gitter.im/spearow/juice

## Benefits+

Juice was built with three concepts in mind: accessibility/simplicity,
performance and portability. We want developers and companies to be able to
run their machine learning applications anywhere: on servers, desktops,
smartphones and embedded devices. Any combination of platform and
computation language (OpenCL, CUDA, etc.) is a first class citizen in Juice.

We coupled portability with simplicity, meaning you can deploy your machine
learning applications to almost any machine and device with no code changes.
Learn more at chapter [4. Backend](./backend.html) or at the
[Coaster Github repository](https://github.com/spearow/juice/tree/master/coaster).

## Contributing

Want to contribute? Awesome!
[We have instructions to help you get started](https://github.com/spearow/juice/blob/master/juice/CONTRIBUTING.md).

Juice has a near real-time collaboration culture, which happens at the [Github
repository](https://github.com/spearow/juice) and on the
[Gitter Channel](https://gitter.im/spearow/juice).

## API Documentation

Alongside this book you can also read the Rust API documentation if
you would like to use Juice as a crate, write a library on top of it or
just want a more low-level overview.

[Rust API documentation](./api-docs.html)

## License

Juice is free for anyone for whatever purpose.
Juice is licensed under either
[Apache License v2.0](https://github.com/spearow/juice/blob/master/juice/LICENSE-APACHE) or,
[MIT license](https://github.com/spearow/juice/blob/master/juice/LICENSE-MIT).
Whatever strikes your fancy.
