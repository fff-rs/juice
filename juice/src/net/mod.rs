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

mod common;
mod config;
mod container;
mod context;
mod descriptor;
mod layer;
mod loss;

use crate::co::{IBackend};
use crate::util::LayerOps;

pub use common::*;
pub use config::*;
pub use container::*;
pub use context::*;
pub use descriptor::*;
pub use layer::*;
pub use loss::*;

// Create layer from a config.
// Takes a partially filled Descriptor, which should have a valid path and fully populated inputs
// (including data_path).
pub fn layer_from_config<B: IBackend + LayerOps<f32> + 'static>(
    descriptor: Descriptor,
    backend: &B,
    config: &LayerConfig,
) -> Box<dyn Layer<B>> {
    match config {
        LayerConfig::Sequential(sequential_config) => {
            Box::new(Sequential::new(descriptor, backend, sequential_config))
        }
        LayerConfig::Linear(linear_config) => Box::new(Linear::new(descriptor, linear_config)),
        LayerConfig::LogSoftmax => Box::new(LogSoftmax::new(descriptor)),
        LayerConfig::NegativeLogLikelihood(nll_config) => {
            Box::new(NegativeLogLikelihood::new(descriptor, nll_config))
        }
    }
}
