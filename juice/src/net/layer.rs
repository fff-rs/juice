use std::fmt::Debug;

use crate::co::{IBackend};
use crate::net::{Context, Descriptor, LayerConfig};
use crate::net::activation::*;
use crate::net::common::*;
use crate::net::container::*;
use crate::net::loss::*;
use crate::util::LayerOps;

/// A generalized layer in a network, performing certain function on inputs producing outputs.
/// Layers be can combined in an acyclic graph forming an ML network that can compute output from
/// inputs and can be "trained" using the backpropagation process.
///
/// Note that a Layer is a more general concept than conventional ML layers and includes:
/// * conventional "layers" like convolutional, fully-connected, dropout, etc;
/// * activation functions like ReLU, softmax, etc;
/// * groups of sublayers.
///
/// Layer can have arbitrary number of inputs, outputs and weights, which are all described in the
/// `Descriptor`. Inputs and outputs declare the shapes of the 'units' of data, which then can
/// be batched according to `Context` settings. The actual shapes of then inputs and outputs are
/// always of the form [N, {unit_shape}] where N is the batch size. 
/// 
/// Number and unit shapes of the inputs are defined by the upstream logic. Number and unit shapes
/// of the outputs are determined by the layer depending on input unit shapes and layer settings.
/// When creating a layer, parent logic passes a partially filled `Descriptor`, containing inputs
/// information. Layer then must fill the outputs of the `Descriptor`.
///
/// It is assumed that weight shapes do not depend on batch size N (as weights are created once and
/// cannot change shape during learning).
pub trait Layer<B: IBackend>: Debug {
    // Computes output given the input(s).
    fn compute_output(&self, backend: &B, context: &mut Context);

    fn compute_gradients(&self, backend: &B, context: &mut Context);

    fn descriptor(&self) -> &Descriptor;

    fn descriptor_mut(&mut self) -> &mut Descriptor;
}

/// Creates layer from a config.
/// Takes a partially filled Descriptor, which should have a valid path and fully populated inputs
/// (including data_path).
/// This is an internal function, typically users should be using net_from_config() instead.
/// TODO: Make it private (for now required for Solver).
pub fn layer_from_config<B: IBackend + LayerOps<f32> + 'static>(
    descriptor: Descriptor,
    config: &LayerConfig,
) -> Box<dyn Layer<B>> {
    match config {
        LayerConfig::Sequential(sequential_config) => {
            Box::new(Sequential::new(descriptor, sequential_config))
        }
        LayerConfig::Fanout(fanout_config) => Box::new(Fanout::new(descriptor, fanout_config)),
        LayerConfig::Linear(linear_config) => Box::new(Linear::new(descriptor, linear_config)),
        LayerConfig::LogSoftmax => Box::new(LogSoftmax::new(descriptor)),
        LayerConfig::Relu => Box::new(Relu::new(descriptor)),
        LayerConfig::Sigmoid => Box::new(Sigmoid::new(descriptor)),
        LayerConfig::NegativeLogLikelihood(nll_config) => {
            Box::new(NegativeLogLikelihood::new(descriptor, nll_config))
        }
        LayerConfig::MeanSquaredError => Box::new(MeanSquaredError::new(descriptor)),
    }
}