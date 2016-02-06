//! Applies the nonlinear Rectified Linear Unit.
//!
//! Non-linearity activation function: y = max(0, x)
//!
//! This is generally the preferred choice over Sigmod or TanH.
//! The max function used in ReLU is usually faster to compute than the exponentiation
//! needed in a Sigmoid layer.

use co::{IBackend,SharedTensor};
use conn::Relu;
use layer::*;
use shared_memory::*;

#[derive(Debug, Copy, Clone)]
/// ReLU Activation Layer
pub struct ReLU;

impl<B: IBackend + Relu<f32>> ILayer<B> for ReLU {
    impl_ilayer_activation!();

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               bottom_data: &[ArcLock<SharedTensor<f32>>],
               _weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               _weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        let btm = bottom_data[0].read().unwrap();
        top_data[0].write().unwrap().resize(btm.desc()).unwrap();
        top_gradient[0].write().unwrap().resize(btm.desc()).unwrap();
    }
}

impl<B: IBackend + Relu<f32>> ComputeOutput<f32, B> for ReLU {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        backend.relu_plain(input_data[0], output_data[0]).unwrap();
    }
}

impl<B: IBackend + Relu<f32>> ComputeInputGradient<f32, B> for ReLU {
    fn compute_input_gradient(&self,
                              backend: &B,
                              _weights: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        backend.relu_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap();
    }
}

impl<B: IBackend + Relu<f32>> ComputeParametersGradient<f32, B> for ReLU {}
