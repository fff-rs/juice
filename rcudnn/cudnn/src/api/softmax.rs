//! Provides the softmax functionality from the CUDA cuDNN API.

use crate::ffi::*;
use crate::{Error, API};

impl API {
    /// Computes an softmax forward function.
    pub fn softmax_forward(
        handle: cudnnHandle_t,
        algorithm: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_softmax_forward(
                handle, algorithm, mode, alpha, src_desc, src_data, beta, dest_desc, dest_data,
            )
        }
    }

    /// Computes an softmax backward function.
    pub fn softmax_backward(
        handle: cudnnHandle_t,
        algorithm: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        src_diff_desc: cudnnTensorDescriptor_t,
        src_diff_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_diff_desc: cudnnTensorDescriptor_t,
        dest_diff_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_softmax_backward(
                handle,
                algorithm,
                mode,
                alpha,
                src_desc,
                src_data,
                src_diff_desc,
                src_diff_data,
                beta,
                dest_diff_desc,
                dest_diff_data,
            )
        }
    }

    unsafe fn ffi_softmax_forward(
        handle: cudnnHandle_t,
        algorithm: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnSoftmaxForward(handle, algorithm, mode, alpha, src_desc, src_data, beta, dest_desc, dest_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("`algorithm` or `mode` are invalid or dimensions or data types of input and output tensor differ or `data_type` or strides of the tensors differ.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute softmax forward.")),
        }
    }

    unsafe fn ffi_softmax_backward(
        handle: cudnnHandle_t,
        algorithm: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        src_diff_desc: cudnnTensorDescriptor_t,
        src_diff_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_diff_desc: cudnnTensorDescriptor_t,
        dest_diff_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnSoftmaxBackward(handle, algorithm, mode, alpha, src_desc, src_data, src_diff_desc, src_diff_data, beta, dest_diff_desc, dest_diff_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("`algorithm` or `mode` are invalid or dimensions or data types of input and output tensor differ or `data_type` or strides of the tensors differ.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute softmax backward.")),
        }
    }
}
