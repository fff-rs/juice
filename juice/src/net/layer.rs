use std::fmt::Debug;

use crate::co::IBackend;
use crate::net::activation::*;
use crate::net::common::*;
use crate::net::container::*;
use crate::net::loss::*;
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

#[derive(Debug, Clone, PartialEq)]
pub enum LayerFromConfigError {
    Sequential(SequentialBadInputOutputError),
}

/// Creates a layer from a config.
/// Takes a partially filled Descriptor, which should have a valid path and inputs.
pub fn layer_from_config<B: IBackend + LayerOps<f32> + 'static>(
    descriptor: Descriptor,
    config: &LayerConfig,
) -> Result<Box<dyn Layer<B>>, LayerFromConfigError> {
    Ok(match config {
        LayerConfig::Linear(cfg) => Box::new(Linear::new(descriptor, cfg)),
        LayerConfig::MeanSquaredError => Box::new(MeanSquaredError::new(descriptor)),
        LayerConfig::NegativeLogLikelihood(cfg) => Box::new(NegativeLogLikelihood::new(descriptor, cfg)),
        LayerConfig::Relu => Box::new(Relu::new(descriptor)),
        LayerConfig::Sequential(cfg) => Box::new(Sequential::new(descriptor, cfg)?),
        LayerConfig::Sigmoid => Box::new(Sigmoid::new(descriptor)),
    })
}

impl From<SequentialBadInputOutputError> for LayerFromConfigError {
    fn from(e: SequentialBadInputOutputError) -> Self {
        LayerFromConfigError::Sequential(e)
    }
}

#[cfg(test)]
pub mod testing {
    use coaster::{Backend, ITensorDesc, Native, SharedTensor};

    use crate::{
        net::{Context, LearnableParamsLink, Network},
        util::{native_backend, write_batch_sample},
    };

    // For floating-point comparisons.
    const EPS: f32 = 0.00001;

    #[derive(Debug)]
    pub struct LayerOutput {
        pub output: SharedTensor<f32>,
    }

    #[derive(Debug)]
    pub struct LayerOutputAndGradients {
        pub output: SharedTensor<f32>,
        pub input_gradient: SharedTensor<f32>,
        pub params_gradients: Vec<SharedTensor<f32>>,
    }

    // Set learnable params to given values. Panics if the overall sizes do not match.
    pub fn set_params(params_link: &LearnableParamsLink, values: &[f32]) {
        let mut params = params_link.borrow_mut();
        assert_eq!(params.data.desc().size(), values.len());
        let params_data = params.data.write_only(native_backend().device()).unwrap();
        params_data.as_mut_slice::<f32>().copy_from_slice(values);
    }

    // Checks tensor equality. Convenience function for unit tests that allows to
    // specify expected tensor as a 2D array. Example:
    //   assert_tensor_eq(tensor, &[[1.0, -2.0]], &[[-3.0, 4.0]]);
    pub fn assert_tensor_eq<Expected: AsRef<[ExpectedRow]>, ExpectedRow: AsRef<[f32]>>(
        tensor: &SharedTensor<f32>,
        expected: Expected,
    ) {
        let expected_rows = expected.as_ref().len();
        let expected_cols = expected.as_ref()[0].as_ref().len();

        // Make sure overall sizes match.
        assert_eq!(expected_rows * expected_cols, tensor.desc().size());

        let backend = native_backend();

        let data = tensor.read(backend.device()).unwrap();
        let slice = data.as_slice::<f32>();
        for i in 0..expected_rows {
            let mut equal = true;
            for j in 0..expected_cols {
                if (slice[i * expected_cols + j] - expected.as_ref()[i].as_ref()[j]).abs() > EPS {
                    equal = false;
                    break;
                }
            }
            if !equal {
                panic!(
                    "Row {} not equal: {:?} vs {:?}",
                    i,
                    &slice[i * expected_cols..(i + 1) * expected_cols],
                    &expected.as_ref()[i].as_ref()
                );
            }
        }
    }

    // Returns network output for a given input which can be given as a 2d array.
    // Used in unit testing. Example:
    //    let result = get_net_output(&net, &[[1.0, -2.0],
    //                                        [-3.0, 4.0]]);
    pub fn get_net_output<Input: AsRef<[InputRow]>, InputRow: AsRef<[f32]>>(
        net: &Network<Backend<Native>>,
        input: Input,
    ) -> LayerOutput {
        let input_rows = input.as_ref().len();
        let input_cols = input.as_ref()[0].as_ref().len();

        // Create input tensor.
        let mut input_tensor = SharedTensor::new(&[input_rows, input_cols]);
        for i in 0..input_rows {
            write_batch_sample(&mut input_tensor, input.as_ref()[i].as_ref(), i);
        }

        // Run the input through the network.
        let backend = native_backend();
        LayerOutput {
            output: net.transform(&backend, &input_tensor),
        }
    }

    // Returns network output and input/params gradients for a given input and output gradient
    // which can be given as a 2d array. Used in unit testing. Example:
    //    let result = get_net_output_and_gradients(&net,
    //                                              &[[1.0, -2.0],
    //                                                [-3.0, 4.0]],
    //                                              &[[0.4, 0.3],
    //                                                [0.1, 0.2]]);
    pub fn get_net_output_and_gradients<
        Input: AsRef<[InputRow]>,
        InputRow: AsRef<[f32]>,
        Output: AsRef<[OutputRow]>,
        OutputRow: AsRef<[f32]>,
    >(
        net: &Network<Backend<Native>>,
        input: Input,
        output_gradient: Output,
    ) -> LayerOutputAndGradients {
        let input_rows = input.as_ref().len();
        let input_cols = input.as_ref()[0].as_ref().len();
        let output_rows = output_gradient.as_ref().len();
        let output_cols = output_gradient.as_ref()[0].as_ref().len();

        // We treat the input and output rows and the batches, so they must be equal.
        assert_eq!(input_rows, output_rows);

        let backend = native_backend();

        // Check that layer output dimensions match the provided output gradient.
        // Again, output unit shape is just [output_cols] since rows are batches.
        assert_eq!(net.top().descriptor().outputs().len(), 1);
        assert_eq!(net.top().descriptor().output(0).unit_shape(), &[output_cols]);

        let mut context = Context::new(input_rows);

        // Create input tensor on the context and fill it.
        let input_tensor = context.acquire_data(net.top().descriptor().input(0));
        for i in 0..input_rows {
            write_batch_sample(&mut input_tensor.borrow_mut(), input.as_ref()[i].as_ref(), i);
        }

        // Compute network output.
        net.top().compute_output(&backend, &mut context);

        // Create output gradient tensor on the context and fill it.
        let output_gradient_tensor = context.acquire_data_gradient(net.top().descriptor().output(0));
        for i in 0..output_rows {
            write_batch_sample(
                &mut output_gradient_tensor.borrow_mut(),
                output_gradient.as_ref()[i].as_ref(),
                i,
            );
        }

        // Make the layer compute the input gradient.
        net.top().compute_gradients(&backend, &mut context);

        // Extract the data from the context and return it.
        LayerOutputAndGradients {
            output: context.take_data(net.top().descriptor().output(0)),
            input_gradient: context.take_data_gradient(net.top().descriptor().input(0)),
            params_gradients: net
                .top()
                .descriptor()
                .params()
                .iter()
                .map(|p| context.take_params_gradient(p))
                .collect(),
        }
    }
}
