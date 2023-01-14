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
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

use coaster::TensorDesc;

use crate::co::{IBackend, ITensorDesc, SharedTensor};
use crate::net::{Context, Descriptor, Layer, LayerError, LearnableParams};
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

pub struct Convolution<B: conn::Convolution<f32>> {
    descriptor: Descriptor,

    config: ConvolutionConfig,

    // Convolution kernel.
    kernel: Rc<RefCell<LearnableParams>>,

    // Backend-specific convolution contexts. Since convolution context depends on full
    // input tensor shape (including batch dimension), we store multiple contexts, one
    // for each batch size ever used before.
    // TODO: Make the ConvolutionContext internally enlarge its workspace when input size
    // changes. This way it won't need to depend on input/output shape when constructing.
    convolution_contexts: RefCell<HashMap<usize, B::CC>>,
}

fn get_input_dimensions(input_shape: &TensorDesc) -> (usize, usize, usize) {
    match input_shape.len() {
        // Input is a [W, H] tensor.
        // TODO: cuDNN always expects [batch_size, C, W, H] input shape. Workaround this?
        2 => unimplemented!("Use [1, W, H] shape for grascale images"),
        // Input is a [C, W, H] tensor.
        3 => (input_shape[0], input_shape[1], input_shape[2]),
        _ => panic!(
            "Can't take an input shape {:?} for convolution layer, expecting either [W, H] or [C, W, H]",
            input_shape
        ),
    }
}

fn get_output_dimension(input_size: usize, padding: usize, stride: usize, kernel_size: usize) -> usize {
    (input_size + padding * 2 - kernel_size) / stride + 1
}

impl<B: conn::Convolution<f32>> Convolution<B> {
    pub fn new(mut descriptor: Descriptor, config: &ConvolutionConfig) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should be only one input.

        // Get the input dimensions.
        let (input_channels, input_width, input_height) = get_input_dimensions(descriptor.input(0).unit_shape());

        // Compute the size of an output feature map(s).
        let output_width = get_output_dimension(input_width, config.padding, config.stride, config.kernel_size);
        let output_height = get_output_dimension(input_height, config.padding, config.stride, config.kernel_size);
        descriptor.add_output(vec![config.feature_maps, output_width, output_height]);

        // Create kernel params tensor.
        let mut kernel_tensor = SharedTensor::<f32>::new(&[
            config.feature_maps,
            input_channels,
            config.kernel_size,
            config.kernel_size,
        ]);
        FillerType::fill_glorot(
            &mut kernel_tensor,
            descriptor.input(0).unit_shape().size(),
            descriptor.output(0).unit_shape().size(),
        );
        let kernel = descriptor.create_params("kernel", kernel_tensor, 1.0);

        Convolution {
            descriptor,
            config: config.clone(),
            kernel,
            convolution_contexts: RefCell::new(HashMap::new()),
        }
    }
}

impl<B: IBackend + LayerOps<f32>> Layer<B> for Convolution<B> {
    fn compute_output(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));

        // If this is the first time compute_output() is called with this batch_size, create a new
        // convolution context for it. Store it in the context cache to reuse later.
        let mut convolution_context_ref = self.convolution_contexts.borrow_mut();
        let mut convolution_context = convolution_context_ref.entry(context.batch_size()).or_insert_with(|| {
            backend
                .new_convolution_context(
                    &input.borrow(),
                    &output.borrow(),
                    &self.kernel.borrow().data,
                    conn::ConvForwardAlgo::Auto,
                    conn::ConvBackwardFilterAlgo::Auto,
                    conn::ConvBackwardDataAlgo::Auto,
                    &[self.config.stride as i32, self.config.stride as i32],
                    &[self.config.padding as i32, self.config.padding as i32],
                )
                .unwrap()
        });

        backend.convolution(
            &self.kernel.borrow().data,
            &input.borrow(),
            &mut output.borrow_mut(),
            &mut convolution_context,
        )?;
        Ok(())
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        let input = context.get_data(self.descriptor.input(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));

        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        let kernel_gradient = context.acquire_params_gradient(self.descriptor.param(0));

        // At this point the convolution context must already exist in the contexts cache
        // (should have been created in the compute_output() function),
        // so we just take it from the cache.
        let mut convolution_context_ref = self.convolution_contexts.borrow_mut();
        let mut convolution_context = convolution_context_ref.get_mut(&context.batch_size()).unwrap();

        // Network error gradient with respect to input data.
        backend.convolution_grad_data(
            &self.kernel.borrow().data,
            &output_gradient.borrow(),
            &mut input_gradient.borrow_mut(),
            convolution_context,
        )?;

        // Network error gradient with respect to the kernel.
        backend.convolution_grad_filter(
            &input.borrow(),
            &mut output_gradient.borrow(),
            &mut kernel_gradient.borrow_mut(),
            &mut convolution_context,
        )?;
        Ok(())
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}

impl<B: conn::Convolution<f32>> Debug for Convolution<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Convolution")
            .field("descriptor", &self.descriptor)
            .field("kernel_shape", self.kernel.borrow().data.desc())
            .finish()
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use coaster::frameworks::cuda::get_cuda_backend;

    use crate::net::{testing::*, Network};

    use super::ConvolutionConfig;

    #[test]
    fn compute() {
        let backend = get_cuda_backend();

        let net = Network::from_config(
            &backend,
            ConvolutionConfig {
                feature_maps: 1,
                padding: 0,
                stride: 1,
                kernel_size: 2,
            },
            &[vec![1, 3, 3]],
        )
        .unwrap();

        //               | 1.0 0.2 |
        // Set kernel to | 0.5 0.1 |.
        set_params(&net.top().descriptor().params()[0], &[1.0, 0.5, 0.2, 0.1]);

        //                                   | 1.0 2.0 3.0 |
        // Apply convolution to input matrix | 4.0 5.0 6.0 |
        //                                   | 7.0 8.0 9.0 |.
        let result = get_net_output(
            &backend,
            &net,
            &create_tensor_3d([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]),
        );

        assert_tensor_eq(&result.output, &create_tensor_3d([[[7.5, 9.3], [12.9, 14.7]]]));
    }

    // TODO: Unit test for compute_gradients().
}
