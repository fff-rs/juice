//! Computes the multinomial logistic loss of the softmax of its bottom Blob.
//!
//! This is conceptually identical to a softmax layer followed by a multinomial
//! logistic loss layer, but provides a more numerically stable gradient.

use co::{IBackend, ITensorDesc, SharedTensor};
use conn;
use layer::*;
use util::native_backend;
use std::f32;
use co::plugin::numeric_helpers::*;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Softmax Loss Layer
pub struct SoftmaxLoss {
    softmax_axis: Option<usize>,
}

impl SoftmaxLoss {
    /// Create a SoftmaxLoss layer from a FullyConnectedConfig.
    pub fn from_config(config: &SoftmaxLossConfig) -> SoftmaxLoss {
        SoftmaxLoss {
            softmax_axis: config.axis,
        }
    }

}

impl ::std::default::Default for SoftmaxLoss {
    fn default() -> SoftmaxLoss {
        SoftmaxLoss {
            softmax_axis: None,
        }
    }
}

impl<B: IBackend + conn::Softmax<f32>> ILayer<B> for SoftmaxLoss {
    impl_ilayer_loss!();

    // fn init(&mut self, backend: Rc<B>) {
    //     self.softmax_axis = Some(1); // TODO: make configurable
    //     // let bottom: ReadBlob = unimplemented!();
    //     // let outer_num = bottom.shape().iter().take(softmax_axis + 1).fold(1, |prod, i| prod * i);
    //     // let inner_num = bottom.shape().iter().skip(softmax_axis + 1).fold(1, |prod, i| prod * i);
    // }
}

impl<B: IBackend + conn::Softmax<f32>> ComputeOutput<f32, B> for SoftmaxLoss {
    fn compute_output(&self,
                      backend: &B,
                      _weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let input_data_0 = input_data[0];
        let label = input_data[1];
        let ref mut probability = output_data[0];

        let _ = backend.softmax_plain(input_data_0, probability);

        let native = native_backend();
        let shared_label = label.get(native.device()).unwrap();
        let memory_label = shared_label.as_native().unwrap();
        let native_label = memory_label.as_slice::<f32>();
        match probability.add_device(native.device()) { _ => probability.sync(native.device()).unwrap() }
        let native_probability = probability.get(native.device()).unwrap().as_native().unwrap().as_slice::<f32>();

        let mut loss = 0f32;
        let outer_num = input_data_0.desc().iter().take(self.softmax_axis.unwrap() + 1).fold(1, |prod, i| prod * i);
        let inner_num = input_data_0.desc().iter().skip(self.softmax_axis.unwrap() + 1).fold(1, |prod, i| prod * i);
        let dim: usize = probability.desc().size() / outer_num;
        for i in 0..(outer_num - 1) {
            for j in 0..(inner_num - 1) {
                let label_value: usize = native_label[i * inner_num + j] as usize;
                loss -= native_probability[i * dim + label_value * inner_num + j].max(f32::MIN).ln();
            }
        }
    }
}

impl<B: IBackend + conn::Softmax<f32>> ComputeInputGradient<f32, B> for SoftmaxLoss {
    fn compute_input_gradient(&self,
                              backend: &B,
                              _weights: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        backend.softmax_grad_plain(output_data[0], output_gradients[0], input_gradients[0]).unwrap();
    }
}

impl<B: IBackend + conn::Softmax<f32>> ComputeParametersGradient<f32, B> for SoftmaxLoss { }

#[derive(Debug, Copy, Clone)]
/// Specifies configuration parameters for a SoftmaxLoss Layer.
pub struct SoftmaxLossConfig {
    /// The axis along which to perform the softmax.
    ///
    /// Preceding dimensions are treated as independent inputs
    ///
    /// Defaults to `1`
    pub axis: Option<usize>,
}

impl SoftmaxLossConfig {
    /// The axis along which to perform the softmax.
    ///
    /// Preceding dimensions are treated as independent inputs
    ///
    /// Defaults to `1`
    pub fn axis(&self) -> usize {
        self.axis.unwrap_or(1)
    }
}
