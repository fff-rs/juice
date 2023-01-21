use std::fmt::Debug;

use thiserror::Error;

use crate::co::IBackend;
use crate::net::activation::*;
use crate::net::common::*;
use crate::net::container::*;
use crate::net::loss::*;
use crate::net::{Context, Descriptor, LayerConfig};
use crate::util::LayerOps;

#[derive(Error, Debug)]
pub enum LayerError {
    #[error("backend error")]
    Backend(Box<dyn std::error::Error>),
    #[error("tensor error")]
    Tensor(Box<dyn std::error::Error>),
}

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
    fn compute_output(&self, backend: &B, context: &mut Context) -> Result<(), LayerError>;

    // Computes the input and weight gradients and stores them in the Context.
    // Invoked during backward pass. Inputs, outputs and output gradients must be already computed
    // and present on the Context. (An output gradient is computed as the input gradient by the
    // downstream layer which uses this output as input.)
    fn compute_gradients(&self, backend: &B, context: &mut Context) -> Result<(), LayerError>;

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
    backend: &B,
    descriptor: Descriptor,
    config: &LayerConfig,
) -> Result<Box<dyn Layer<B>>, LayerFromConfigError> {
    Ok(match config {
        LayerConfig::Convolution(cfg) => Box::new(Convolution::new(descriptor, cfg)),
        LayerConfig::Dropout(cfg) => Box::new(Dropout::new(backend, descriptor, cfg)),
        LayerConfig::Linear(cfg) => Box::new(Linear::new(descriptor, cfg)),
        LayerConfig::LogSoftmax => Box::new(LogSoftmax::new(descriptor)),
        LayerConfig::MeanSquaredError => Box::new(MeanSquaredError::new(descriptor)),
        LayerConfig::NegativeLogLikelihood => Box::new(NegativeLogLikelihood::new(descriptor)),
        LayerConfig::Pooling(cfg) => Box::new(Pooling::new(backend, descriptor, cfg)),
        LayerConfig::Relu => Box::new(Relu::new(descriptor)),
        LayerConfig::Sequential(cfg) => Box::new(Sequential::new(backend, descriptor, cfg)?),
        LayerConfig::Sigmoid => Box::new(Sigmoid::new(descriptor)),
        LayerConfig::Softmax => Box::new(Softmax::new(descriptor)),
    })
}

impl From<::coaster::error::Error> for LayerError {
    fn from(e: ::coaster::error::Error) -> Self {
        LayerError::Backend(Box::new(e))
    }
}

impl From<::coaster::tensor::Error> for LayerError {
    fn from(e: ::coaster::tensor::Error) -> Self {
        LayerError::Tensor(Box::new(e))
    }
}

impl From<SequentialBadInputOutputError> for LayerFromConfigError {
    fn from(e: SequentialBadInputOutputError) -> Self {
        LayerFromConfigError::Sequential(e)
    }
}

#[cfg(test)]
pub mod testing {
    use coaster::{frameworks::native::get_native_backend, IBackend, ITensorDesc, SharedTensor};

    use crate::{
        net::{Context, LearnableParamsLink, Network},
        util::{format_tensor, native_backend, LayerOps},
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

    // Convenience method for creating a single-dimension tensor from f32 slices.
    pub fn create_tensor_1d<Dim1: AsRef<[f32]>>(data: Dim1) -> SharedTensor<f32> {
        let dim1 = data.as_ref().len();
        let mut tensor = SharedTensor::new(&[dim1]);
        let slice = tensor
            .write_only(native_backend().device())
            .unwrap()
            .as_mut_slice::<f32>();
        slice.copy_from_slice(data.as_ref());
        tensor
    }

    // Convenience method for creating a two-dimension tensor from f32 slices.
    pub fn create_tensor_2d<Dim2: AsRef<[Dim1]>, Dim1: AsRef<[f32]>>(data: Dim2) -> SharedTensor<f32> {
        let dim2 = data.as_ref().len();
        let dim1 = data.as_ref()[0].as_ref().len();
        let mut tensor = SharedTensor::new(&[dim2, dim1]);
        let slice = tensor
            .write_only(native_backend().device())
            .unwrap()
            .as_mut_slice::<f32>();
        for i2 in 0..dim2 {
            slice[i2 * dim1..(i2 + 1) * dim1].copy_from_slice(data.as_ref()[i2].as_ref());
        }
        tensor
    }

    // Convenience method for creating a three-dimension tensor from f32 slices.
    pub fn create_tensor_3d<Dim3: AsRef<[Dim2]>, Dim2: AsRef<[Dim1]>, Dim1: AsRef<[f32]>>(
        data: Dim3,
    ) -> SharedTensor<f32> {
        let dim3 = data.as_ref().len();
        let dim2 = data.as_ref()[0].as_ref().len();
        let dim1 = data.as_ref()[0].as_ref()[0].as_ref().len();
        let mut tensor = SharedTensor::new(&[dim3, dim2, dim1]);
        let slice = tensor
            .write_only(native_backend().device())
            .unwrap()
            .as_mut_slice::<f32>();
        for i3 in 0..dim3 {
            let offset = i3 * dim2 * dim1;
            for i2 in 0..dim2 {
                slice[offset + i2 * dim1..offset + (i2 + 1) * dim1]
                    .copy_from_slice(data.as_ref()[i3].as_ref()[i2].as_ref());
            }
        }
        tensor
    }

    // Checks tensor equality and prints both if they differ.
    pub fn assert_tensor_eq(tensor: &SharedTensor<f32>, expected: &SharedTensor<f32>) {
        assert_eq!(
            tensor.desc().size(),
            expected.desc().size(),
            "Tensor overall sizes differ, expected {} got {}",
            expected.desc().size(),
            tensor.desc().size()
        );

        let backend = get_native_backend();

        let t2_data = tensor.read(backend.device()).unwrap().as_slice::<f32>();
        let e2_data = expected.read(backend.device()).unwrap().as_slice::<f32>();

        const TOLERANCE: f32 = 1e-5;
        let equal = t2_data
            .iter()
            .zip(e2_data.iter())
            .find(|(v1, v2)| (**v1 - **v2).abs() > TOLERANCE)
            .is_none();
        assert!(
            equal,
            "Tensor \n{} doesn't match expected \n{}",
            format_tensor(tensor),
            format_tensor(expected)
        );
    }

    // Returns network output for a given input which can be given as a 2d array.
    // First dimension in the input tensor is assumed to be the batch size.
    // Used in unit testing. Example:
    //    let result = get_net_output(&net, &create_tensor_2d([[1.0, -2.0],
    //                                                        [-3.0, 4.0]]));
    pub fn get_net_output<B: IBackend + LayerOps<f32> + 'static>(
        backend: &B,
        net: &Network<B>,
        input: &SharedTensor<f32>,
    ) -> LayerOutput {
        // Run the input through the network.
        LayerOutput {
            output: net.transform(backend, input).unwrap(),
        }
    }

    // Returns network output and input/params gradients for a given input and output gradient
    // which can be given as a 2d array. First dimension in the input tensor is assumed to be the
    // batch size. Used in unit testing. Example:
    //    let result = get_net_output_and_gradients(&net, &create_tensor_2d([[1.0, -2.0],
    //                                                                      [-3.0, 4.0]]),
    //                                                    &create_tensor_2d([[0.4, 0.3],
    //                                                                      [0.1, 0.2]]);
    pub fn get_net_output_and_gradients<B: IBackend + LayerOps<f32> + 'static>(
        backend: &B,
        net: &Network<B>,
        input: &SharedTensor<f32>,
        output_gradient: &SharedTensor<f32>,
    ) -> LayerOutputAndGradients {
        let native_backend = get_native_backend();
        let batch_size = input.desc()[0];

        // Check that layer output size match the provided output gradient.
        assert_eq!(net.top().descriptor().outputs().len(), 1);
        assert_eq!(
            net.top().descriptor().output(0).unit_shape().size() * batch_size,
            output_gradient.desc().size()
        );

        let mut context = Context::new(batch_size);

        // Copy input data into the context.
        {
            let context_inputs = context.acquire_data(net.top().descriptor().input(0));
            assert_eq!(context_inputs.borrow().desc().size(), input.desc().size());
            backend.copy(&input, &mut context_inputs.borrow_mut()).unwrap();
        }

        // Compute network output.
        net.top().compute_output(backend, &mut context).unwrap();

        // Copy output gradient into the context.
        {
            let output_gradient_tensor_ref = context.acquire_data_gradient(net.top().descriptor().output(0));

            let context_output_gradient = context.acquire_data_gradient(net.top().descriptor().output(0));
            assert_eq!(
                context_output_gradient.borrow().desc().size(),
                output_gradient.desc().size()
            );
            backend
                .copy(&output_gradient, &mut context_output_gradient.borrow_mut())
                .unwrap();
        }

        // Make the layer compute the input gradient.
        net.top().compute_gradients(&backend, &mut context).unwrap();

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
