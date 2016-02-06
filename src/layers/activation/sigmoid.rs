//! Applies the nonlinear Log-Sigmoid function.
//!
//! Non-linearity activation function: y = (1 + e^(-x))^(-1)
//!
//! A classic choice in neural networks.
//! But you might consider using ReLu as an alternative.
//!
//! ReLu, compared to Sigmoid
//!
//! * reduces the likelyhood of vanishing gradients
//! * increases the likelyhood of a more beneficial sparse representation
//! * can be computed faster
//! * is therefore the most popular activation function in DNNs as of this
//! writing (2015).
use co::{IBackend, SharedTensor};
use conn;
use layer::*;
use shared_memory::*;

#[derive(Debug, Copy, Clone)]
/// Sigmoid Activation Layer
pub struct Sigmoid;

impl<B: IBackend + conn::Sigmoid<f32>> ILayer<B> for Sigmoid {
    impl_ilayer_activation!();

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               bottom_data: &[ArcLock<SharedTensor<f32>>],
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        let btm = bottom_data[0].read().unwrap();
        top_data[0].write().unwrap().resize(btm.desc()).unwrap();
        top_gradient[0].write().unwrap().resize(btm.desc()).unwrap();
    }
}

impl<B: IBackend + conn::Sigmoid<f32>> ComputeOutput<f32, B> for Sigmoid {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        backend.sigmoid_plain(input_data[0], output_data[0]).unwrap();
    }
}

impl<B: IBackend + conn::Sigmoid<f32>> ComputeInputGradient<f32, B> for Sigmoid {
    fn compute_input_gradient(&self,
                              backend: &B,
                              _weights: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        backend.sigmoid_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0]).unwrap();
    }
}

impl<B: IBackend + conn::Sigmoid<f32>> ComputeParametersGradient<f32, B> for Sigmoid {}
