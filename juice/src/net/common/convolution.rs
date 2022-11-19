//! Convolution layer applies a 2D convolution operation (with a convolution kernel) to an input
//! tensor to produce a feature map. Convolution kernel coefficients are learnable params; the
//! goal of training is learn a kernel that can produce features that are useful for downstream
//! leayers.
//!
//! Producing N different feature maps requires maintaining and learning N independent kernels.
//! For convenience, instead of instantiating N parallel convolution layers, this implementation
//! supports producing multiple feature maps in a single layer; the underlying kernels are still
//! learned independently.
//!
//! Padding and stride are supported as scalars. Same padding and stride values will be applied
//! to both horizontal and vertical dimensions (see below).
//!
//! Input tensor can have one of the following shapes (ignoring the batch dimension):
//! 1. A 2D tensor with dimensions [W, H]. This usually represents a grayscale image of WxH size.
//! 2. A 3D tensor with dimensions [C, W, H]. This usually represents a color picture with C
//!    color channels (which can be RGB, HSV etc) with each channel having WxH size.
//!
//! In both cases above, output will have shape [N, OW, OH] where N is the number of feature maps
//! and OW and OH are horizontal and vertical dimensions after applying padding and stride.

use std::cell::RefCell;
use std::rc::Rc;

use coaster::TensorDesc;

use crate::co::{IBackend, ITensorDesc, SharedTensor};
use crate::coblas::transpose::Transpose;
use crate::net::{Context, Descriptor, Layer, LearnableParams};
use crate::util::LayerOps;
use crate::weight::FillerType;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConvolutionConfig {
    // Number of output feature maps.
    pub feature_maps: usize,
    // Padding and stride that are applied to both dimensions of the incoming 2D array.
    pub padding: usize,
    pub stride: usize,
    // Filter size. The resulting convolution filter will have shape [kernel_size, kernel_size]
    // for single-channel inputs and [C, kernel_size, kernel_size] for inputs with C channels.
    pub kernel_size: usize,
}

#[derive(Debug)]
pub struct Convolution<B: conn::Convolution<f32>> {
    descriptor: Descriptor,

    // Convolution kernel.
    kernel: Rc<RefCell<LearnableParams>>,

    // Backend-specific convolution context.
    convolution_context: B::CC,
}

fn get_input_dimensions(input_shape: &TensorDesc) -> (usize, usize, usize) {
    unimplemented!();
}

fn get_output_size(input_size: usize, padding: usize, stride: usize, kernel_size: usize) -> usize {
    unimplemented!();
}

impl<B: conn::Convolution<f32>> Convolution<B> {
    pub fn new(backend: &B, mut descriptor: Descriptor, config: &ConvolutionConfig) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should be only one input.

        // Get the input dimensions.
        let (input_channels, input_width, input_height) = get_input_dimensions(descriptor.input(0).unit_shape());

        // Compute the size of an output feature map.
        let output_width = get_output_size(input_width, config.padding, config.stride, config.kernel_size);
        let output_height = get_output_size(input_height, config.padding, config.stride, config.kernel_size);
        descriptor.add_output(vec![config.feature_maps, output_width, output_height]);

        // Create kernel params tensor.
        let mut kernel_tensor = SharedTensor::<f32>::new(&[input_channels, config.kernel_size, config.kernel_size]);
        FillerType::fill_glorot(
            &mut kernel_params,
            descriptor.input(0).unit_shape().size(),
            descriptor.output(0).unit_shape().size(),
        );

        let context = backend
            .new_convolution_context(
                &inp,
                &output_data,
                &mut kernel_tensor,
                conn::ConvForwardAlgo::Auto,
                conn::ConvBackwardFilterAlgo::Auto,
                conn::ConvBackwardDataAlgo::Auto,
                &stride,
                &padding,
            )
            .unwrap();

        Convolution {
            descriptor,
            kernel: descriptor.create_params("kernel", kernel_tensor, 1.0),
        }
    }
}

impl<B: IBackend + LayerOps<f32>> Layer<B> for Convolution {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));

        let mut ones_tensor = SharedTensor::<f32>::new(&[context.batch_size(), 1]);
        FillerType::fill_constant(&mut ones_tensor, 1f32);

        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &ones_tensor,
                Transpose::NoTrans,
                &self.bias.borrow().data,
                &self.zero,
                &mut output.borrow_mut(),
            )
            .unwrap();

        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &input.borrow(),
                Transpose::Trans,
                &self.weight.borrow().data,
                &self.one,
                &mut output.borrow_mut(),
            )
            .unwrap();
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));

        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        let weights_gradient = context.acquire_params_gradient(self.descriptor.param(0));
        let bias_gradient = context.acquire_params_gradient(self.descriptor.param(1));

        // Network error gradient with respect to input data.
        // dE/dx = dE/dy * df/dx = dE/dy * w.
        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &output_gradient.borrow(),
                Transpose::NoTrans,
                &self.weight.borrow().data,
                &self.zero,
                &mut input_gradient.borrow_mut(),
            )
            .unwrap();

        // Network error gradient with respect to weights.
        // dE/dw = dE/dy * df/dw = dE/dy * x.
        backend
            .gemm(
                &self.one,
                Transpose::Trans,
                &output_gradient.borrow(),
                Transpose::NoTrans,
                &input.borrow(),
                &self.zero,
                &mut weights_gradient.borrow_mut(),
            )
            .unwrap();

        // Network error gradient with respect to bias.
        // dE/dw = dE/dy * df/db = dE/dy * [1] = dE/dy.
        let mut ones_row = SharedTensor::new(&vec![1, context.batch_size()]);
        FillerType::fill_constant(&mut ones_row, 1.0);
        backend
            .gemm(
                &self.one,
                Transpose::NoTrans,
                &ones_row,
                Transpose::NoTrans,
                &output_gradient.borrow(),
                &self.zero,
                &mut bias_gradient.borrow_mut(),
            )
            .unwrap();
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}

#[cfg(test)]
mod tests {
    use crate::net::{layer::testing::*, LayerConfig, Network};

    use super::ConvolutionConfig;

    #[test]
    fn compute() {
        let net = Network::from_config(
            LayerConfig::Convolution(ConvolutionConfig { output_size: 2 }),
            &[vec![3]],
        )
        .unwrap();

        // Set params such that layer becomes this:
        //            |1 4|
        // |x1 x2 x3| |2 5| + |0.1 0.2|
        //            |3 6|
        // Note that weights are stored transposed.
        set_params(&net.top().descriptor().params()[0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        set_params(&net.top().descriptor().params()[1], &[0.1, 0.2]);

        let result = get_net_output(&net, &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        assert_tensor_eq(&result.output, &[[14.1, 32.2], [32.1, 77.2]]);
    }

    #[test]
    fn compute_gradients() {
        let net = Network::from_config(
            LayerConfig::Convolution(ConvolutionConfig { output_size: 2 }),
            &[vec![3]],
        )
        .unwrap();

        // Set params such that layer becomes this:
        //            |1 4|
        // |x1 x2 x3| |2 5| + |0.1 0.2|
        //            |3 6|
        // Note that weights are stored transposed.
        set_params(&net.top().descriptor().params()[0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        set_params(&net.top().descriptor().params()[1], &[0.1, 0.2]);

        // Output gradient contains a single non-zero item at pos 0,0.
        {
            let result = get_net_output_and_gradients(
                &net,
                &[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
                &[[1.0, 0.0], [0.0, 0.0]],
            );
            assert_tensor_eq(&result.input_gradient, &[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(&result.params_gradients[0], &[[0.01, 0.02, 0.03], [0.0, 0.0, 0.0]]);
            assert_tensor_eq(&result.params_gradients[1], &[[1.0, 0.0]]);
        }

        // Output gradient contains a single non-zero item at pos 0,1.
        {
            let result = get_net_output_and_gradients(
                &net,
                &[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
                &[[0.0, 1.0], [0.0, 0.0]],
            );
            assert_tensor_eq(&result.input_gradient, &[[4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]);
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(&result.params_gradients[0], &[[0.0, 0.0, 0.0], [0.01, 0.02, 0.03]]);
            assert_tensor_eq(&result.params_gradients[1], &[[0.0, 1.0]]);
        }

        // Output gradient contains a single non-zero item at pos 1,0.
        {
            let result = get_net_output_and_gradients(
                &net,
                &[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
                &[[0.0, 0.0], [1.0, 0.0]],
            );
            assert_tensor_eq(&result.input_gradient, &[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(&result.params_gradients[0], &[[0.04, 0.05, 0.06], [0.0, 0.0, 0.0]]);
            assert_tensor_eq(&result.params_gradients[1], &[[1.0, 0.0]]);
        }

        // Output gradient contains a single non-zero item at pos 1,1.
        {
            let result = get_net_output_and_gradients(
                &net,
                &[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
                &[[0.0, 0.0], [0.0, 1.0]],
            );
            assert_tensor_eq(&result.input_gradient, &[[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]);
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(&result.params_gradients[0], &[[0.0, 0.0, 0.0], [0.04, 0.05, 0.06]]);
            assert_tensor_eq(&result.params_gradients[1], &[[0.0, 1.0]]);
        }

        // Output gradient contains all 1s.
        {
            let result = get_net_output_and_gradients(
                &net,
                &[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],
                &[[1.0, 1.0], [1.0, 1.0]],
            );
            assert_tensor_eq(&result.input_gradient, &[[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]);
            assert_eq!(result.params_gradients.len(), 2);
            assert_tensor_eq(&result.params_gradients[0], &[[0.05, 0.07, 0.09], [0.05, 0.07, 0.09]]);
            assert_tensor_eq(&result.params_gradients[1], &[[2.0, 2.0]]);
        }
    }
}
