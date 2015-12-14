//! Describes utility functionality for CUDA cuDNN.
use std::marker::PhantomData;
use ffi::*;

#[derive(Debug, Copy, Clone)]
/// Provides a convenient interface for forward/backwar functionality.
pub enum Direction {
    /// Forward computation
    Fr,
    /// Backward computation
    Bc
}

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface to access cuDNN's convolution parameters,
/// `algo` and `workspace` and `workspace_size_in_bytes`.
///
/// You woudn't use this struct yourself, but rather obtain it through `Cudnn.init_convolution()`.
pub struct ConvolutionConfig {
    forward_algo: cudnnConvolutionFwdAlgo_t,
    backward_algo: cudnnConvolutionBwdDataAlgo_t,
    forward_workspace: *mut ::libc::c_void,
    forward_workspace_size: usize,
    backward_workspace: *mut ::libc::c_void,
    backward_workspace_size: usize,
}

impl ConvolutionConfig {
    /// Returns a new ConvolutionConfig
    pub fn new(
        algo_fwd: cudnnConvolutionFwdAlgo_t,
        workspace_fwd: *mut ::libc::c_void,
        workspace_size_fwd: usize,
        algo_bwd: cudnnConvolutionBwdDataAlgo_t,
        workspace_bwd: *mut ::libc::c_void,
        workspace_size_bwd: usize,
    ) -> ConvolutionConfig {
        ConvolutionConfig {
            forward_algo: algo_fwd,
            backward_algo: algo_bwd,
            forward_workspace: workspace_fwd,
            forward_workspace_size: workspace_size_fwd,
            backward_workspace: workspace_bwd,
            backward_workspace_size: workspace_size_bwd,
        }
    }

    /// Returns `forward_algo`.
    pub fn forward_algo(&self) -> &cudnnConvolutionFwdAlgo_t {
        &self.forward_algo
    }

    /// Returns `forward_workspace`.
    pub fn forward_workspace(&self) -> &*mut ::libc::c_void {
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
    pub fn backward_workspace(&self) -> &*mut ::libc::c_void {
        &self.backward_workspace
    }

    /// Returns `backward_workspace_size`.
    pub fn backward_workspace_size(&self) -> &usize {
        &self.backward_workspace_size
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

impl ScalParams<f32> {
    /// Provides default values for ScalParams<f32>.
    pub fn default() -> ScalParams<f32> {
        let alpha_ptr: *const ::libc::c_void = *&[1.0f32].as_ptr() as *const ::libc::c_void;
        let beta_ptr: *const ::libc::c_void = *&[0.0f32].as_ptr() as *const ::libc::c_void;
        ScalParams {
            a: alpha_ptr,
            b: beta_ptr,
            scal_type: PhantomData,
        }
    }
}

impl ScalParams<f64> {
    /// Provides default values for ScalParams<f64>.
    pub fn default() -> ScalParams<f64> {
        let alpha_ptr: *const ::libc::c_void = *&[1.0f64].as_ptr() as *const ::libc::c_void;
        let beta_ptr: *const ::libc::c_void = *&[0.0f64].as_ptr() as *const ::libc::c_void;
        ScalParams {
            a: alpha_ptr,
            b: beta_ptr,
            scal_type: PhantomData,
        }
    }
}
