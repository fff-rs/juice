//! Applies Softmax to the top Blob
use co::{IBackend, SharedTensor};
use conn;
use layer::*;

#[derive(Debug, Copy, Clone)]
/// Softmax Layer
pub struct Softmax;

impl<B: IBackend + conn::Softmax<f32>> ILayer<B> for Softmax {
    impl_ilayer_common!();
}

impl<B: IBackend + conn::Softmax<f32>> ComputeOutput<f32, B> for Softmax {
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        backend.softmax_plain(input_data[0], output_data[0]).unwrap();
    }
}

impl<B: IBackend + conn::Softmax<f32>> ComputeInputGradient<f32, B> for Softmax {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        backend.softmax_grad_plain(output_data[0], output_gradients[0], input_gradients[0]).unwrap();
    }
}

impl<B: IBackend + conn::Softmax<f32>> ComputeParametersGradient<f32, B> for Softmax { }
