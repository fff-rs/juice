use std::fmt::Debug;

use crate::co::IBackend;
use crate::net::activation::*;
use crate::net::container::Sequential;
use crate::net::{Context, Descriptor, LayerConfig};
use crate::util::LayerOps;

/// A generalized layer in a network, performing certain function on inputs producing outputs.
/// Layers be can combined in an acyclic graph forming a network that can compute output from
/// inputs and can be "trained" using the backpropagation process.
///
/// Note that a Layer is a more general concept than conventional ML layers and includes:
/// * conventional layers like convolutional, fully-connected, dropout, etc;
/// * activation functions like ReLU, softmax, etc;
/// * groups of sublayers.
///
/// Layer can have arbitrary number of inputs, outputs and weights, which are all described in the
/// `Descriptor`. Inputs and outputs declare the shapes of the 'units' of data, which then can
/// be batched according to `Context` settings. The actual shapes of the inputs and outputs are
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
    // Computes output given the input(s) and stores them in the Context.
    // Invoked during forward pass. Inputs must be already computed and present on the Context
    // (will panic otherwise).
    fn compute_output(&self, backend: &B, context: &mut Context);

    // Computes the input and weight gradients and stores them in the Context.
    // Invoked during backward pass. Inputs, outputs and output gradients must be already computed
    // and present on the Context. (An output gradient is computed as the input gradient by the
    // downstream layer which uses this output as input.)
    fn compute_gradients(&self, backend: &B, context: &mut Context);

    // Returns the immutable Descriptor ref.
    fn descriptor(&self) -> &Descriptor;

    // Returns the mutable Descriptor ref. Typically used during construction by the parent logic
    // to connect outputs of this layer to the inputs of the next one.
    fn descriptor_mut(&mut self) -> &mut Descriptor;
}

/// Creates a layer from a config.
/// Takes a partially filled Descriptor, which should have a valid path and inputs.
pub fn layer_from_config<B: IBackend + LayerOps<f32> + 'static>(
    descriptor: Descriptor,
    config: &LayerConfig,
) -> Box<dyn Layer<B>> {
    match config {
        LayerConfig::Sequential(cfg) => Box::new(Sequential::new(descriptor, cfg)),
        LayerConfig::Relu => Box::new(Relu::new(descriptor)),
    }
}
