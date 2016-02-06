//! Acts as a fully-connected layer.

use std::rc::Rc;
use co::backend::IBackend;
use co::tensor::SharedTensor;
use coblas::transpose::Transpose;
use coblas::plugin::*;
use layer::*;
use shared_memory::*;
use util::{native_scalar, LayerOps};

#[derive(Debug)]
/// FullyConnected Layer
pub struct FullyConnected {
    num_output: usize,

    axis: usize,

    one: SharedTensor<f32>,
    zero: SharedTensor<f32>,
}

impl FullyConnected {
    /// Create a FullyConnected layer from a FullyConnectedConfig.
    pub fn from_config(config: &FullyConnectedConfig) -> FullyConnected {
        let one = native_scalar(1f32);
        let zero = native_scalar(0f32);

        FullyConnected {
            num_output: config.num_output,

            axis: config.axis(),

            one: one,
            zero: zero,
        }
    }
}

impl ::std::default::Default for FullyConnected {
    fn default() -> FullyConnected {
        let config = FullyConnectedConfig {
            num_output: 10,

            axis: None,
        };

        Self::from_config(&config)
    }
}

impl<B: IBackend + LayerOps<f32>> ILayer<B> for FullyConnected {
    impl_ilayer_common!();

    fn init(&mut self, backend: Rc<B>) {
        let device = <B as IBackend>::device(&backend);
        self.one.add_device(device).unwrap();
        self.one.sync(device).unwrap();
        self.zero.add_device(device).unwrap();
        self.zero.sync(device).unwrap();
    }

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               bottom_data: &[ArcLock<SharedTensor<f32>>],
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        // reshape top
        let btm = bottom_data[0].read().unwrap();
        let mut top_data = top_data[0].write().unwrap();
        let mut top_shape = btm.desc()[0..self.axis].to_vec();
        top_shape.push(self.num_output);
        top_data.resize(&top_shape).unwrap();
        top_gradient[0].write().unwrap().resize(&top_shape).unwrap();
        // reshape weight
        let m = btm.desc().iter().skip(1).fold(1, |prod, i| prod * i);
        let weight_shape = vec![self.num_output, m];
        // TODO: change weight creation to not require this
        if let Some(weight) = weights_data.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
        }
        if let Some(weight) = weights_gradient.get(0) {
            weight.write().unwrap().resize(&weight_shape).unwrap();
        }
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeOutput<f32, B> for FullyConnected {
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let input_data_i = input_data[0];
        let weight = weights[0];
        let ref mut output_data_i = output_data[0];

        backend.gemm_plain(&self.one, Transpose::NoTrans, input_data_i, Transpose::Trans, weight, &self.zero, output_data_i).unwrap();
        let has_bias_term = false; // TODO: implement bias term
        if has_bias_term {
            let bias_multiplier = unimplemented!();
            let bias_data = unimplemented!();
            backend.gemm_plain(&self.one, Transpose::NoTrans, bias_multiplier, Transpose::NoTrans, bias_data, &self.one, output_data_i).unwrap();
        }
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeInputGradient<f32, B> for FullyConnected {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        let output_gradient = output_gradients[0];
        let weight_data = weights[0];
        let ref mut input_gradient = input_gradients[0];
        // Gradient with respect to input data
        backend.gemm_plain(&self.one, Transpose::NoTrans, output_gradient, Transpose::NoTrans, weight_data, &self.zero, input_gradient).unwrap();
    }
}

impl<B: IBackend + LayerOps<f32>> ComputeParametersGradient<f32, B> for FullyConnected {
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   output_data: &[&SharedTensor<f32>],
                                   output_gradients: &[&SharedTensor<f32>],
                                   input_data: &[&SharedTensor<f32>],
                                   parameters_gradients: &mut [&mut SharedTensor<f32>]) {
        // TODO: implement gradient w.r.t weights and bias
        // if (this->param_propagate_down_[0]) {
        //     const Dtype* top_diff = top[0]->gpu_diff();
        //     const Dtype* bottom_data = bottom[0]->gpu_data();
        //     // Gradient with respect to weight
        //     caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        //         top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
        // }
        // if (bias_term_ && this->param_propagate_down_[1]) {
        //     const Dtype* top_diff = top[0]->gpu_diff();
        //     // Gradient with respect to bias
        //     caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        //         bias_multiplier_.gpu_data(), (Dtype)1.,
        //         this->blobs_[1]->mutable_gpu_diff());
        // }
    }
}

#[derive(Debug, Copy, Clone)]
/// Specifies configuration parameters for a FullyConnected Layer.
pub struct FullyConnectedConfig {
    /// The number of output values
    pub num_output: usize,
    /// The first axis in the inner product operation.
    ///
    /// Preceding dimensions are treated as independent inputs
    ///
    /// Defaults to `1`
    pub axis: Option<usize>,
}

impl FullyConnectedConfig {
    /// The first axis in the inner product operation.
    ///
    /// Preceding dimensions are treated as independent inputs
    ///
    /// Defaults to `1`
    pub fn axis(&self) -> usize {
        self.axis.unwrap_or(1)
    }
}
