use std::fmt::Debug;

use crate::co::{IBackend};
use crate::net::{Context, Descriptor};

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
    fn compute_output(&self, context: &mut Context<B>);

    fn compute_gradients(&self, context: &mut Context<B>);

    fn descriptor(&self) -> &Descriptor;

    fn descriptor_mut(&mut self) -> &mut Descriptor;
}
