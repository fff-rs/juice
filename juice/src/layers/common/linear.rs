//! Applies a linear transformation to the input data `y = a * x + b`
//!
//! The variables are:
//!
//! - `y`: output value
//! - `a`: weight (a trainable weight in a neural network)
//! - `x`: input value
//! - `b`: bias (only for Backends with the `coblas::plugin::Copy trait`)
//!
//! ## Input Data
//!
//! The input can either have one or two dimensions:
//!
//! - If the input has one dimension the transformation will just be applied to the input data.
//! - If the input has two dimensions **the first dimension is treated as batch size** (`N`)
//!   and the transformation will be applied to every vector in the second dimension, using the
//!   same weights and biases.
//!
//! In the context of convolutional neural networks this layer is also
//! called a "fully-connected layer" if it is used at the end of the network.

use crate::capnp_util::*;
use crate::co::backend::IBackend;
use crate::co::tensor::SharedTensor;
use crate::coblas::transpose::Transpose;
use crate::juice_capnp::linear_config as capnp_config;
use crate::layer::*;
use crate::util::{native_scalar, ArcLock, LayerOps};
use crate::weight::FillerType;

#[derive(Debug)]
/// Linear Layer
pub struct Linear {
    output_size: usize,

    one: SharedTensor<f32>,
    zero: SharedTensor<f32>,
    ones_row: SharedTensor<f32>,
}

impl Linear {
    /// Create a Linear layer from a LinearConfig.
    pub fn from_config(config: &LinearConfig) -> Linear {
        let one = native_scalar(1f32);
        let zero = native_scalar(0f32);
        let ones_row = SharedTensor::new(&vec![1]);

        Linear {
            output_size: config.output_size,

            one,
            zero,
            ones_row,
        }
    }

    // Calculates the input size by skipping the batch size.
    fn calculate_input_size(input_shape: &[usize]) -> usize {
        input_shape.iter().skip(1).fold(1, |prod, i| prod * i)
    }

    fn calculate_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let n = input_shape[0]; // batch size
        vec![n, self.output_size]
    }

    fn calculate_weight_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let m = Self::calculate_input_size(input_shape);
        vec![self.output_size, m]
    }
}

impl<B: IBackend + LayerOps<f32>> ILayer<B> for Linear {
    fn auto_weight_blobs(&self) -> bool {
        true
    }

    fn reshape(
        &mut self,
        backend: ::std::rc::Rc<B>,
        input_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        input_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
        output_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
    ) {
        let input = input_data[0].read().unwrap();
        let batch_size = input.desc()[0];
        // reshape top
        let output_shape = self.calculate_output_shape(input.desc());
        output_data[0].write().unwrap().resize(&output_shape).unwrap();
        output_gradient[0].write().unwrap().resize(&output_shape).unwrap();
        // reshape weight
        let weight_shape = self.calculate_weight_shape(input.desc());
        // TODO: change weight creation to not require this
        if let Some(weight) = weights_data.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
            let filler = FillerType::Glorot {
                input_size: Self::calculate_input_size(input.desc()),
                output_size: self.output_size,
            };
            filler.fill(&mut weight.write().unwrap());
        }
        if let Some(weight) = weights_gradient.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
        }

        // Fill the bias
        if let Some(weight) = weights_data.get(1) {
            weight.write().unwrap().resize(&(1, self.output_size)).unwrap();
            // Weight Initialisation for bias is typically a constant, and a suitable initialisation
            // is stated in https://cs231n.github.io/neural-networks-2/#init for non-LSTM types.
            let initialisation_constant = rand::random::<f32>();
            let filler = FillerType::Constant {
                value: initialisation_constant * (2.0 / initialisation_constant).sqrt(),
            };
            filler.fill(&mut weight.write().unwrap());
        }
        if let Some(weight) = weights_gradient.get(1) {
            weight.write().unwrap().resize(&(1, self.output_size)).unwrap();
        }

        // Reshape the column of 1s which is used to compute bias gradient.
        self.ones_row.resize(&vec![1, batch_size]).unwrap();
        FillerType::fill_constant(&mut self.ones_row, 1.0);
    }

    fn exact_num_output_blobs(&self) -> Option<usize> {
        Some(1)
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeOutput<f32, B> for Linear {
    /// Basically, x has the shape (k, n) where k is the batch size. Given W with shape (m, n) where
    /// m is output vector length, we compute the output with the formula xW^T which will give us a
    /// matrix of size (k, m) with the outputs.
    fn compute_output(
        &self,
        backend: &B,
        weights: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        output_data: &mut [&mut SharedTensor<f32>],
    ) {
        let mut ones_tensor = SharedTensor::<f32>::new(&[input_data[0].desc().as_slice()[0], 1]);
        FillerType::fill_constant(&mut ones_tensor, 1f32);
        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &ones_tensor,
                Transpose::NoTrans,
                weights[1],
                &self.zero,
                output_data[0],
            )
            .unwrap();

        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                input_data[0],
                Transpose::Trans,
                weights[0],
                &self.one,
                output_data[0],
            )
            .unwrap();
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeInputGradient<f32, B> for Linear {
    /// Since we have row vectors instead of columns, xW^T = (Wx^T)^T. Take the derivative with
    /// respect to x^T (gives us a column vector of dimension (n, 1)), we get d((Wx^T)^T)/d(x^T) =
    /// W^T of dims (n, m). In backpropagation with column vectors, we would take W^T * output_grad,
    /// and in terms of row vectors, that would be output_grad^T * W which produces a vector of
    /// dims (1, n)
    fn compute_input_gradient(
        &self,
        backend: &B,
        weights_data: &[&SharedTensor<f32>],
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        input_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        // Gradient with respect to input data
        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                output_gradients[0],
                Transpose::NoTrans,
                weights_data[0],
                &self.zero,
                input_gradients[0],
            )
            .unwrap();
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeParametersGradient<f32, B> for Linear {
    fn compute_parameters_gradient(
        &self,
        backend: &B,
        output_data: &[&SharedTensor<f32>],
        output_gradients: &[&SharedTensor<f32>],
        input_data: &[&SharedTensor<f32>],
        parameters_gradients: &mut [&mut SharedTensor<f32>],
    ) {
        // gradient w.r.t. weights
        backend
            .gemm(
                &self.one,
                Transpose::Trans,
                output_gradients[0],
                Transpose::NoTrans,
                input_data[0],
                &self.zero,
                parameters_gradients[0],
            )
            .unwrap();

        // gradient w.r.t bias
        // The gradient of vector b of length n to itself is the I_n identity matrix,
        // so multiply output_gradient[0] by a 1-column.
        // Note that we instead multiply a one-row by output_gradient[0], since doing it
        // in the opposite order causes gemm() implementation to miscalculate dimensions.
        // TODO: Fix this.
        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &self.ones_row,
                Transpose::NoTrans,
                output_gradients[0],
                &self.zero,
                parameters_gradients[1],
            )
            .unwrap();
    }
}

impl ::std::default::Default for Linear {
    fn default() -> Linear {
        let config = LinearConfig { output_size: 10 };

        Self::from_config(&config)
    }
}

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Specifies configuration parameters for a Linear Layer.
pub struct LinearConfig {
    /// The number of output values
    pub output_size: usize,
}

impl<'a> CapnpWrite<'a> for LinearConfig {
    type Builder = capnp_config::Builder<'a>;

    /// Write the LinearConfig into a capnp message.
    fn write_capnp(&self, builder: &mut Self::Builder) {
        builder.reborrow().set_output_size(self.output_size as u64);
    }
}

impl<'a> CapnpRead<'a> for LinearConfig {
    type Reader = capnp_config::Reader<'a>;

    fn read_capnp(reader: Self::Reader) -> Self {
        let output_size = reader.get_output_size() as usize;

        LinearConfig {
            output_size: output_size,
        }
    }
}

impl Into<LayerType> for LinearConfig {
    fn into(self) -> LayerType {
        LayerType::Linear(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::co::tensor::SharedTensor;
    use crate::layer::{ComputeInputGradient, ComputeOutput, ComputeParametersGradient};
    use crate::layers::{Linear, LinearConfig};
    use crate::util::native_backend;

    fn get_sample_w() -> &'static [f32] {
        [
            1f32, 0f32, 3f32, 0f32, 1.5f32, 4f32, 2f32, 0f32, 0f32, 2f32, 1.5f32, 4f32,
        ]
        .as_ref()
    }

    fn get_sample_x() -> &'static [f32] {
        [1f32, 2f32, 3f32, 4f32].as_ref()
    }

    fn get_sample_b() -> &'static [f32] {
        [-1f32, 1f32, 0f32].as_ref()
    }

    fn get_sample_output_gradient() -> &'static [f32] {
        [-1f32, 0.5f32, 0.2f32].as_ref()
    }

    #[test]
    fn forward_pass_test() {
        let ref config = LinearConfig { output_size: 3 };
        let layer = Linear::from_config(config);
        let backend = native_backend();

        let ref w_shape = (3, 4);
        let ref x_shape = (1, 4);
        let ref output_shape = (1, 3);
        let b_shape = output_shape;

        let mut w = SharedTensor::<f32>::new(w_shape);
        let mut x = SharedTensor::<f32>::new(x_shape);
        let mut b = SharedTensor::<f32>::new(b_shape);

        w.write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_w());
        x.write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_x());
        b.write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_b());

        let mut output = SharedTensor::<f32>::new(output_shape);

        layer.compute_output(&backend, &[&w, &b], &[&x], &mut [&mut output]);

        let result_slice: &[f32] = output.read(backend.device()).unwrap().as_slice();
        assert_eq!(result_slice, &[9f32, 16.5f32, 24.5f32])
    }

    #[test]
    fn input_gradient_test() {
        let ref config = LinearConfig { output_size: 3 };
        let layer = Linear::from_config(config);
        let backend = native_backend();

        let ref w_shape = (3, 4);
        let ref x_shape = (1, 4);
        let ref output_shape = (1, 3);
        let b_shape = output_shape;

        let mut w = SharedTensor::<f32>::new(w_shape);
        let mut x = SharedTensor::<f32>::new(x_shape);
        let mut b = SharedTensor::<f32>::new(b_shape);

        w.write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_w());
        x.write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_x());
        b.write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_b());

        let mut input_gradient = SharedTensor::<f32>::new(x_shape);
        let mut output_gradient = SharedTensor::<f32>::new(output_shape);
        output_gradient
            .write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_output_gradient());
        // The output_data tensor doesn't really matter since it's not used.
        let output_data = SharedTensor::<f32>::new(&(1, 1));

        layer.compute_input_gradient(
            &backend,
            &[&w, &b],
            &[&output_data],
            &[&output_gradient],
            &[&x],
            &mut [&mut input_gradient],
        );

        let result_slice: &[f32] = input_gradient.read(backend.device()).unwrap().as_slice();
        assert_eq!(result_slice, &[-0.25f32, 2.4f32, -1.7f32, 0.8f32]);
    }

    #[test]
    fn parameter_gradient_test() {
        let ref config = LinearConfig { output_size: 3 };
        let layer = Linear::from_config(config);
        let backend = native_backend();

        let ref w_shape = (3, 4);
        let ref x_shape = (1, 4);
        let ref output_shape = (1, 3);
        let b_shape = output_shape;

        let mut w_grad = SharedTensor::<f32>::new(w_shape);
        let mut x = SharedTensor::<f32>::new(x_shape);
        let mut b_grad = SharedTensor::<f32>::new(b_shape);

        x.write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_x());

        let input_gradient = SharedTensor::<f32>::new(x_shape);
        let mut output_gradient = SharedTensor::<f32>::new(output_shape);
        output_gradient
            .write_only(backend.device())
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(get_sample_output_gradient());
        // The output_data tensor doesn't really matter since it's not used.
        let output_data = SharedTensor::<f32>::new(&(1, 1));

        layer.compute_parameters_gradient(
            &backend,
            &[&output_data],
            &[&output_gradient],
            &[&x],
            &mut [&mut w_grad, &mut b_grad],
        );

        let w_grad_result: &[f32] = w_grad.read(backend.device()).unwrap().as_slice();
        let b_grad_result: &[f32] = b_grad.read(backend.device()).unwrap().as_slice();

        assert_eq!(
            w_grad_result,
            &[-1f32, -2f32, -3f32, -4f32, 0.5f32, 1f32, 1.5f32, 2f32, 0.2f32, 0.4f32, 0.6f32, 0.8f32]
        );
        assert_eq!(b_grad_result, &[-1f32, 0.5f32, 0.2f32]);
    }
}
