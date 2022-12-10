//! Pooling layer applies a 2D pooling operation to a 2D input tensor using a sliding square window.
//! The pooling operation can be either:
//! * Max: resulting value will be the maximum value of all values in the window;
//! * Avg: resulting value will be the average value of all values in the window.
//!
//! This layer is typically used right after the convolution layer to downsample the feature
//! maps produced by convolution layers. Since convolution layer typically produces several
//! feature maps, this layer supports input tensors of the following shapes (ignoring the batch
//! dimension):
//! 1. A 2D tensor with dimensions [W, H]. This implies a single incoming feature map. The output
//!    will have shape [OW, OH] where OW and OH are horizontal and vertical dimensions after
//!    applying padding and stride.
//! 2. A 3D tensor with dimensions [C, W, H]. This implies C independent incoming feature maps.
//!    The output will have shape [C, OW, OH] where each individual subtensor [OW, OH] will
//!    contain pooling result from the corresponding incoming feature map.

use std::fmt::{Debug, Formatter};

use coaster::TensorDesc;

use crate::co::IBackend;
use crate::net::{Context, Descriptor, Layer};
use crate::util::LayerOps;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// The different modes of pooling that can be calculated.
pub enum PoolingMode {
    /// The maximum value inside the pooling window will be used as result.
    Max,
    /// The average of all values inside the pooling window will be used as result.
    Average,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoolingConfig {
    /// The pooling mode to use.
    pub mode: PoolingMode,
    // Padding and stride that are applied to both dimensions of the incoming 2D array.
    pub padding: usize,
    pub stride: usize,
    /// Pooling window size. The window will have [window_size, window_size] shape.
    pub window_size: usize,
}

pub struct Pooling<B: conn::Pooling<f32>> {
    descriptor: Descriptor,

    mode: PoolingMode,
    backend_config: B::CPOOL,
}

fn get_input_dimensions(input_shape: &TensorDesc) -> (usize, usize, usize) {
    match input_shape.len() {
        // Input is a [W, H] tensor.
        2 => unimplemented!("Use [1, W, H] shape for standalone 2D tensors"),
        // Input is a [C, W, H] tensor
        3 => (input_shape[0], input_shape[1], input_shape[2]),
        _ => panic!(
            "Can't take an input shape {:?} for pooling layer, expecting either [W, H] or [C, W, H]",
            input_shape
        ),
    }
}

fn get_output_dimension(input_size: usize, padding: usize, stride: usize, kernel_size: usize) -> usize {
    (input_size + padding * 2 - kernel_size) / stride + 1
}

impl Default for PoolingMode {
    fn default() -> Self {
        PoolingMode::Max
    }
}

impl<B: conn::Pooling<f32>> Pooling<B> {
    pub fn new(backend: &B, mut descriptor: Descriptor, config: &PoolingConfig) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should be only one input.

        // Get the input dimensions.
        let (input_channels, input_width, input_height) = get_input_dimensions(descriptor.input(0).unit_shape());

        // Compute the size of an output feature map(s).
        let output_width = get_output_dimension(input_width, config.padding, config.stride, config.window_size);
        let output_height = get_output_dimension(input_height, config.padding, config.stride, config.window_size);
        descriptor.add_output(vec![input_channels, output_width, output_height]);

        let backend_config = backend
            .new_pooling_config(
                &[config.window_size as i32, config.window_size as i32],
                &[config.stride as i32, config.stride as i32],
                &[config.padding as i32, config.padding as i32],
            )
            .unwrap();

        Pooling {
            descriptor,
            mode: config.mode,
            backend_config,
        }
    }
}

impl<B: IBackend + LayerOps<f32>> Layer<B> for Pooling<B> {
    fn compute_output(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));

        match self.mode {
            PoolingMode::Max => backend
                .pooling_max(&input.borrow(), &mut output.borrow_mut(), &self.backend_config)
                .unwrap(),
            PoolingMode::Average => backend
                .pooling_avg(&input.borrow(), &mut output.borrow_mut(), &self.backend_config)
                .unwrap(),
        }
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));

        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));

        match self.mode {
            PoolingMode::Max => backend
                .pooling_max_grad(
                    &output.borrow(),
                    &output_gradient.borrow(),
                    &input.borrow(),
                    &mut input_gradient.borrow_mut(),
                    &self.backend_config,
                )
                .unwrap(),
            PoolingMode::Average => backend
                .pooling_avg_grad(
                    &output.borrow(),
                    &output_gradient.borrow(),
                    &input.borrow(),
                    &mut input_gradient.borrow_mut(),
                    &self.backend_config,
                )
                .unwrap(),
        }
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}

impl<B: conn::Pooling<f32>> Debug for Pooling<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pooling")
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use coaster::frameworks::cuda::get_cuda_backend;

    use crate::net::{testing::*, Network};

    use super::{PoolingConfig, PoolingMode};

    #[test]
    fn compute() {
        let backend = get_cuda_backend();

        let net = Network::from_config(
            &backend,
            PoolingConfig {
                mode: PoolingMode::Max,
                padding: 0,
                stride: 1,
                window_size: 2,
            },
            &[vec![1, 3, 3]],
        )
        .unwrap();

        //                                   | 1.0 2.0 3.0 |
        // Apply pooling to an input matrix  | 4.0 5.0 6.0 |
        //                                   | 7.0 8.0 9.0 |.
        let result = get_net_output(
            &backend,
            &net,
            &create_tensor_3d([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]),
        );

        //                         | 5.0 6.0 |
        // Should result in matrix | 8.0 9.0 |.
        assert_tensor_eq(&result.output, &create_tensor_3d([[[5.0, 6.0], [8.0, 9.0]]]));
    }

    #[test]
    fn compute_gradients() {
        let backend = get_cuda_backend();

        let net = Network::from_config(
            &backend,
            PoolingConfig {
                mode: PoolingMode::Max,
                padding: 0,
                stride: 1,
                window_size: 2,
            },
            &[vec![1, 3, 3]],
        )
        .unwrap();

        // Output gradient contains a single non-zero item at pos 0,0.
        // Input gradient should have a single 1 for the maximum element
        // in the window [0..2,0..2], which is element at 1, 1.
        {
            let result = get_net_output_and_gradients(
                &backend,
                &net,
                &create_tensor_3d([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]),
                &create_tensor_3d([[[1.0, 0.0], [0.0, 0.0]]]),
            );
            assert_tensor_eq(
                &result.input_gradient,
                &create_tensor_3d([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]),
            );
        }
    }
}
