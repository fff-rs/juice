//! Describes utility functionality for CUDA cuDNN.

#[derive(Debug, Copy, Clone)]
/// Provides a convenient interface for forward/backwar functionality.
pub enum Direction {
    /// Forward computation
    Fr,
    /// Backward computation
    Bc
}

#[allow(missing_debug_implementations, missing_copy_implementations)]
/// Provides a convenient interface for cuDNN's scaling parameters `alpha` and `beta`.
///
/// Scaling paramarters lend the source value with prior value in the destination
/// tensor as follows: dstValue = alpha[0]*srcValue + beta[0]*priorDstValue. When beta[0] is
/// zero, the output is not read and can contain any uninitialized data (including NaN). The
/// storage data type for alpha[0], beta[0] is float for HALF and SINGLE tensors, and double
/// for DOUBLE tensors. These parameters are passed using a host memory pointer.
/// For improved performance it is advised to use beta[0] = 0.0. Use a non-zero value for
/// beta[0] only when blending with prior values stored in the output tensor is needed.
pub struct ScalParams {
    /// Alpha
    pub a: *const ::libc::c_void,
    /// Beta
    pub b: *const ::libc::c_void,
}

impl Default for ScalParams {
    fn default() -> ScalParams {
        ScalParams {
            a: ::std::ptr::null(),
            b: ::std::ptr::null(),
        }
    }
}
