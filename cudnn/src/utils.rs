//! Describes utility functionality for CUDA cuDNN.

use super::{
    ActivationDescriptor, ConvolutionDescriptor, DropoutDescriptor, FilterDescriptor,
    NormalizationDescriptor, PoolingDescriptor,
};
use cuda::CudaDeviceMemory;

use ffi::*;

use num::traits::*;

#[derive(Debug, Copy, Clone)]
/// Defines the available data types for the CUDA cuDNN data representation.
pub enum DataType {
    /// F32
    Float,
    /// F64
    Double,
    /// F16 (no native Rust support yet)
    Half,
}

/// CuDnn type info for generic use.
pub trait DataTypeInfo {
    /// Mostly internal.
    fn cudnn_data_type() -> DataType;
}
impl DataTypeInfo for f32 {
    fn cudnn_data_type() -> DataType {
        DataType::Float
    }
}
impl DataTypeInfo for f64 {
    fn cudnn_data_type() -> DataType {
        DataType::Double
    }
}
// TODO f16

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface to access cuDNN's convolution parameters,
/// `algo` and `workspace` and `workspace_size_in_bytes`.
///
/// You woudn't use this struct yourself, but rather obtain it through `Cudnn.init_convolution()`.
pub struct ConvolutionConfig {
    forward_algo: cudnnConvolutionFwdAlgo_t,
    backward_filter_algo: cudnnConvolutionBwdFilterAlgo_t,
    backward_data_algo: cudnnConvolutionBwdDataAlgo_t,
    forward_workspace_size: usize,
    backward_filter_workspace_size: usize,
    backward_data_workspace_size: usize,
    conv_desc: ConvolutionDescriptor,
    filter_desc: FilterDescriptor,
}

impl ConvolutionConfig {
    /// Returns a new ConvolutionConfig
    pub fn new(
        algo_fwd: cudnnConvolutionFwdAlgo_t,
        workspace_size_fwd: usize,
        algo_filter_bwd: cudnnConvolutionBwdFilterAlgo_t,
        workspace_filter_size_bwd: usize,
        algo_data_bwd: cudnnConvolutionBwdDataAlgo_t,
        workspace_data_size_bwd: usize,
        conv_desc: ConvolutionDescriptor,
        filter_desc: FilterDescriptor,
    ) -> ConvolutionConfig {
        ConvolutionConfig {
            forward_algo: algo_fwd,
            forward_workspace_size: workspace_size_fwd,
            backward_filter_algo: algo_filter_bwd,
            backward_filter_workspace_size: workspace_filter_size_bwd,
            backward_data_algo: algo_data_bwd,
            backward_data_workspace_size: workspace_data_size_bwd,
            conv_desc: conv_desc,
            filter_desc: filter_desc,
        }
    }

    /// Returns the largest workspace size out of the three.
    ///
    /// Useful for creating a shared workspace.
    pub fn largest_workspace_size(&self) -> &usize {
        if self.backward_data_workspace_size() >= self.backward_filter_workspace_size()
            && self.backward_data_workspace_size() >= self.forward_workspace_size()
        {
            self.backward_data_workspace_size()
        } else if self.backward_filter_workspace_size() >= self.backward_data_workspace_size()
            && self.backward_filter_workspace_size() >= self.forward_workspace_size()
        {
            self.backward_filter_workspace_size()
        } else {
            self.forward_workspace_size()
        }
    }

    /// Returns `forward_algo`.
    pub fn forward_algo(&self) -> &cudnnConvolutionFwdAlgo_t {
        &self.forward_algo
    }

    /// Returns `forward_workspace_size`.
    pub fn forward_workspace_size(&self) -> &usize {
        &self.forward_workspace_size
    }

    /// Returns `backward_filter_algo`.
    pub fn backward_filter_algo(&self) -> &cudnnConvolutionBwdFilterAlgo_t {
        &self.backward_filter_algo
    }

    /// Returns `backward_filter_workspace_size`.
    pub fn backward_filter_workspace_size(&self) -> &usize {
        &self.backward_filter_workspace_size
    }

    /// Returns `backward_data_algo`.
    pub fn backward_data_algo(&self) -> &cudnnConvolutionBwdDataAlgo_t {
        &self.backward_data_algo
    }

    /// Returns `backward_data_workspace_size`.
    pub fn backward_data_workspace_size(&self) -> &usize {
        &self.backward_data_workspace_size
    }

    /// Returns `conv_desc`.
    pub fn conv_desc(&self) -> &ConvolutionDescriptor {
        &self.conv_desc
    }

    /// Returns `filter_desc`.
    pub fn filter_desc(&self) -> &FilterDescriptor {
        &self.filter_desc
    }
}

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface to access cuDNN's Normalization (LRN) Descriptor.
///
/// You woudn't use this struct yourself, but rather obtain it through `Cudnn.init_normalization()`.
pub struct NormalizationConfig {
    lrn_desc: NormalizationDescriptor,
}

impl NormalizationConfig {
    /// Returns a new LRN Config.
    pub fn new(lrn_desc: NormalizationDescriptor) -> NormalizationConfig {
        NormalizationConfig { lrn_desc: lrn_desc }
    }

    /// Returns `lrn_desc`.
    pub fn lrn_desc(&self) -> &NormalizationDescriptor {
        &self.lrn_desc
    }
}

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface to access cuDNN's Pooling Descriptor.
///
/// You woudn't use this struct yourself, but rather obtain it through `Cudnn.init_pooling()`.
pub struct PoolingConfig {
    pooling_avg_desc: PoolingDescriptor,
    pooling_max_desc: PoolingDescriptor,
}

impl PoolingConfig {
    /// Returns a new PoolingConfig.
    pub fn new(
        pooling_avg_desc: PoolingDescriptor,
        pooling_max_desc: PoolingDescriptor,
    ) -> PoolingConfig {
        PoolingConfig {
            pooling_avg_desc: pooling_avg_desc,
            pooling_max_desc: pooling_max_desc,
        }
    }

    /// Returns `pooling_avg_desc`.
    pub fn pooling_avg_desc(&self) -> &PoolingDescriptor {
        &self.pooling_avg_desc
    }

    /// Returns `pooling_max_desc`.
    pub fn pooling_max_desc(&self) -> &PoolingDescriptor {
        &self.pooling_max_desc
    }
}

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface to access cuDNN's Activation Descriptor.
///
/// You woudn't use this struct yourself, but rather obtain it through `Cudnn.init_activation()`.
pub struct ActivationConfig {
    activation_sigmoid_desc: ActivationDescriptor,
    activation_relu_desc: ActivationDescriptor,
    activation_clipped_relu_desc: ActivationDescriptor,
    activation_tanh_desc: ActivationDescriptor,
}

impl ActivationConfig {
    /// Returns a new ActivationConfig.
    pub fn new(
        activation_sigmoid_desc: ActivationDescriptor,
        activation_relu_desc: ActivationDescriptor,
        activation_clipped_relu_desc: ActivationDescriptor,
        activation_tanh_desc: ActivationDescriptor,
    ) -> ActivationConfig {
        ActivationConfig {
            activation_sigmoid_desc: activation_sigmoid_desc,
            activation_relu_desc: activation_relu_desc,
            activation_clipped_relu_desc: activation_clipped_relu_desc,
            activation_tanh_desc: activation_tanh_desc,
        }
    }

    /// Returns `activation_sigmoid_desc`.
    pub fn activation_sigmoid_desc(&self) -> &ActivationDescriptor {
        &self.activation_sigmoid_desc
    }
    /// Returns `activation_relu_desc`.
    pub fn activation_relu_desc(&self) -> &ActivationDescriptor {
        &self.activation_relu_desc
    }
    /// Returns `activation_clipped_relu_desc`.
    pub fn activation_clipped_relu_desc(&self) -> &ActivationDescriptor {
        &self.activation_clipped_relu_desc
    }
    /// Returns `activation_tanh_desc`.
    pub fn activation_tanh_desc(&self) -> &ActivationDescriptor {
        &self.activation_tanh_desc
    }
}

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface to access cuDNN's Dropout Descriptor.
///
/// You woudn't use this struct yourself, but rather obtain it through `Cudnn.init_dropout()`.
pub struct DropoutConfig {
    dropout_desc: DropoutDescriptor,
    reserve_space: CudaDeviceMemory,
}

impl DropoutConfig {
    /// Returns a new DropoutConfig.
    pub fn new(dropout_desc: DropoutDescriptor, reserve: CudaDeviceMemory) -> DropoutConfig {
        DropoutConfig {
            dropout_desc: dropout_desc,
            reserve_space: reserve,
        }
    }
    /// Returns `dropout_desc`.
    pub fn dropout_desc(&self) -> &DropoutDescriptor {
        &self.dropout_desc
    }

    /// Returns the reserved space ``.
    pub fn reserved_space(&self) -> &CudaDeviceMemory {
        &self.reserve_space
    }
}

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface for cuDNN's scaling parameters `alpha` and `beta`.
///
/// Scaling paramarters lend the source value with prior value in the destination
/// tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue. When beta[0] is
/// zero, the output is not read and can contain any uninitialized data (including NaN). The
/// storage data type for alpha[0], beta[0] is float for HALF and SINGLE tensors, and double
/// for DOUBLE tensors. These parameters are passed using a host memory pointer.
///
/// For improved performance it is advised to use beta[0] = 0.0. Use a non-zero value for
/// beta[0] only when blending with prior values stored in the output tensor is needed.
pub struct ScalParams<T>
where
    T: Float + DataTypeInfo,
{
    /// Alpha
    pub a: T,
    /// Beta
    pub b: T,
}

impl<T> Default for ScalParams<T>
where
    T: Float + Zero + One + DataTypeInfo,
{
    /// Provides default values for ScalParams<f32>.
    fn default() -> ScalParams<T> {
        ScalParams {
            a: One::one(),
            b: Zero::zero(),
        }
    }
}
