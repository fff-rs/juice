//! Representation of a neural network.
//!
//! Network consists of 2 parts:
//! 1. Static layer configuration and connections between them. Each layer takes data as input,
//!    performs certain operation which produced the output. Layers can be combined into
//!    hierarchical structures via special container layers. Configuration of each layer, as
//!    well as connections between them is static, i.e. it cannot change in runtime. The shape
//!    of data 'units' in inputs/outputs is also fixed, but the batch size (batch comprises
//!    several data 'units') is not. Layer inputs/outputs as well as their connections is
//!    captured in a `Descriptor`, which also contains information about learnable params.
//! 2. Dynamic state representing (partial) data flow through the network. When doing the forward
//!    pass through the net (converting inputs to outputs), all the intermediate state (data
//!    buffers passed between layers) is saved in a `Context`, which allows reuse of this data
//!    when doing the backpropagation step (intermediate data for backpropagation, for example,
//!    gradients, is also saved in the `Context`). Context also defines the batch size for this
//!    particular instance of exercising the network, which allows to use different batch sizes
//!    for different use cases (for example simultaneous learning with batches and using the net
//!    for producing outputs by setting batch size to 1).

mod activation;
mod common;
mod config;
mod container;
mod context;
mod descriptor;
mod layer;
mod loss;
mod network;

pub use config::*;
pub use context::*;
pub use descriptor::*;
pub use layer::*;
pub use network::*;
