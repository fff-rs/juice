//! Describes utility functionality for CUDA cuDNN.
use super::{ConvolutionDescriptor, NormalizationDescriptor, FilterDescriptor, PoolingDescriptor};
use ffi::*;
use std::marker::PhantomData;
use co::frameworks::cuda::Memory;

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

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface to access cuDNN's convolution parameters,
/// `algo` and `workspace` and `workspace_size_in_bytes`.
///
/// You woudn't use this struct yourself, but rather obtain it through `Cudnn.init_convolution()`.
pub struct ConvolutionConfig {
    forward_algo: cudnnConvolutionFwdAlgo_t,
    backward_algo: cudnnConvolutionBwdDataAlgo_t,
    forward_workspace: Memory,
    forward_workspace_size: usize,
    backward_workspace: Memory,
    backward_workspace_size: usize,
    conv_desc: ConvolutionDescriptor,
    filter_desc: FilterDescriptor,
    filter_data: Memory,
}

impl ConvolutionConfig {
    /// Returns a new ConvolutionConfig
    pub fn new(
        algo_fwd: cudnnConvolutionFwdAlgo_t,
        workspace_fwd: Memory,
        workspace_size_fwd: usize,
        algo_bwd: cudnnConvolutionBwdDataAlgo_t,
        workspace_bwd: Memory,
        workspace_size_bwd: usize,
        conv_desc: ConvolutionDescriptor,
        filter_desc: FilterDescriptor,
        filter_data: Memory,
    ) -> ConvolutionConfig {
        ConvolutionConfig {
            forward_algo: algo_fwd,
            backward_algo: algo_bwd,
            forward_workspace: workspace_fwd,
            forward_workspace_size: workspace_size_fwd,
            backward_workspace: workspace_bwd,
            backward_workspace_size: workspace_size_bwd,
            conv_desc: conv_desc,
            filter_desc: filter_desc,
            filter_data: filter_data,
        }
    }

    /// Returns `forward_algo`.
    pub fn forward_algo(&self) -> &cudnnConvolutionFwdAlgo_t {
        &self.forward_algo
    }

    /// Returns `forward_workspace`.
    pub fn forward_workspace(&self) -> &Memory {
        &self.forward_workspace
    }

    /// Returns `forward_workspace_size`.
    pub fn forward_workspace_size(&self) -> &usize {
        &self.forward_workspace_size
    }

    /// Returns `backward_algo`.
    pub fn backward_algo(&self) -> &cudnnConvolutionBwdDataAlgo_t {
        &self.backward_algo
    }

    /// Returns `backward_workspace`.
    pub fn backward_workspace(&self) -> &Memory {
        &self.backward_workspace
    }

    /// Returns `backward_workspace_size`.
    pub fn backward_workspace_size(&self) -> &usize {
        &self.backward_workspace_size
    }

    /// Returns `conv_desc`.
    pub fn conv_desc(&self) -> &ConvolutionDescriptor {
        &self.conv_desc
    }

    /// Returns `filter_desc`.
    pub fn filter_desc(&self) -> &FilterDescriptor {
        &self.filter_desc
    }

    /// Returns `filter_data`.
    pub fn filter_data(&self) -> &Memory {
        &self.filter_data
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
        NormalizationConfig {
            lrn_desc: lrn_desc,
        }
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
pub struct ScalParams<T> {
    /// Alpha
    pub a: *const ::libc::c_void,
    /// Beta
    pub b: *const ::libc::c_void,
    scal_type: PhantomData<T>,
}

impl Default for ScalParams<f32> {
    /// Provides default values for ScalParams<f32>.
    fn default() -> ScalParams<f32> {
        let alpha_ptr: *const ::libc::c_void = *&[1.0f32].as_ptr() as *const ::libc::c_void;
        let beta_ptr: *const ::libc::c_void = *&[0.0f32].as_ptr() as *const ::libc::c_void;
        ScalParams {
            a: alpha_ptr,
            b: beta_ptr,
            scal_type: PhantomData,
        }
    }
}

impl Default for ScalParams<f64> {
    /// Provides default values for ScalParams<f64>.
    fn default() -> ScalParams<f64> {
        let alpha_ptr: *const ::libc::c_void = *&[1.0f64].as_ptr() as *const ::libc::c_void;
        let beta_ptr: *const ::libc::c_void = *&[0.0f64].as_ptr() as *const ::libc::c_void;
        ScalParams {
            a: alpha_ptr,
            b: beta_ptr,
            scal_type: PhantomData,
        }
    }
}
