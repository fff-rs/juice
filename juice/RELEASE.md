# Announcing Juice 0.2

We are happy to announce today the release of Juice 0.2 on which we have been
working on for the last weeks. Juice is a modular, performant, portable
Machine Intelligence Framework.
It is the Hacker's Machine Intelligence Framework, developed by software
engineers.

You can [install Juice 0.2][install] and [run examples][examples], including
popular Deep Neural Networks like Alexnet, Overfeat, VGG and more.

## What's in Juice 0.2

The release was mostly about finding an efficient and clean architecture,
catching up with the performance level of other Machine Learning Frameworks. It
shares concepts from the brilliant work done by the people behind Torch,
Tensorflow, Caffe, Rust and numerous research papers. We have several large
features under development, Juice 0.2 gives us the platform to go on exploring
new territory with Juice 0.3.

### Performance

Juice 0.2 is one of the fastest Machine Intelligence Frameworks that exist
today. Rust was a big help in developing the entire platform over the course of
a few months. We achieved a very efficient GPU utilization and oriented our
architecture close to Torch and achieved the distribution capabilities of
Tensorflow, on a lower abstraction level. More information in the
following sections. 

More Benchmarks and comparisons, including Memory utilization, can be found on
[Deep Learning Benchmarks][deep-learning-benchmarks-website].

### Portability

Juice 0.2 uses [Coaster][coaster] for training and running models on
CPUs, GPUs, FPGAs, etc. with OpenCL or CUDA or other Computation Languages, on
various machines and operating systems, without the need to adapt your code what
so ever. This makes deployment of models to servers, desktops, smartphones and
later embedded devices very convenient.

With that abstraction and separation of algorithm representation and execution,
we gain a nice Framework for distributed model execution, without relying
on a symbolic, data-flow graph model like Tensorflow, which introduces
performance and development overhead concerns.

### Architecture

Juice 0.2 replaces special `Network` objects with container layers
like the `Sequential` layer. Where previously all weights were stored centrally
by the Network, each Layer is now responsible for managing its own weights.
This allows for more flexibility in expressing different network architectures.
It also enables better programmatic generation of networks by nesting container
layers where each container represents a common pattern in neural networks,
e.g. Convolution, Pooling and ReLU following each other.

### Contributors for Juice 0.2

We had 9 individual contributors, which made Juice 0.2 possible. Thank you so
much for your contribution, when Juice wasn't even executable, yet. And thank you
for everyone who took the time to engage with us on [Gitter][gitter-juice] and
Github.

* Maximilian Goisser ([@hobofan](https://twitter.com/hobofan))
* Michael Hirn ([@mjhirn](https://twitter.com/mjhirn))
* Ewan Higgs ([ehiggs](https://github.com/ehiggs))
* Florian Gilcher ([@argorak](https://twitter.com/Argorak))
* Paul Dib ([pdib](https://github.com/pdib))
* David Irvine ([dirvine](https://github.com/dirvine))
* Pascal Hertleif ([killercup](https://github.com/killercup))
* Kyle Schmit ([kschmit90](https://github.com/kschmit90))
* Sébastien Lerique ([wehlutyk](https://github.com/wehlutyk))

<div align="center">
  <p>
    <a href="https://spearow.io">More about Juice and Spearow</a>
  </p>
</div>

[install]: https://github.com/spearow/juice#getting-started
[examples]: https://github.com/spearow/juice#examples
[coaster]: https://github.com/spearow/coaster
[deep-learning-benchmarks-website]: http://spearow.io/deep-learning-benchmarks
[gitter-juice]: https://gitter.im/spearow/juice
