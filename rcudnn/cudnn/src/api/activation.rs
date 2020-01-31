//! Provides the activation functionality from the CUDA cuDNN API.

use crate::ffi::*;
use crate::{Error, API};

impl API {
    /// Create a generic CUDA cuDNN ActivationDescriptor.
    pub fn create_activation_descriptor() -> Result<cudnnActivationDescriptor_t, Error> {
        unsafe { API::ffi_create_activation_descriptor() }
    }

    /// Destroys a CUDA cuDNN Activation Descriptor.
    ///
    /// Should be called when freeing a CUDA::Descriptor to not trash up the CUDA device.
    pub fn destroy_activation_descriptor(desc: cudnnActivationDescriptor_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy_activation_descriptor(desc) }
    }

    /// Initializes a generic CUDA cuDNN Activation Descriptor with specific properties.
    pub fn set_activation_descriptor(
        desc: cudnnActivationDescriptor_t,
        mode: cudnnActivationMode_t,
        relu_nan_opt: cudnnNanPropagation_t,
        relu_ceiling: f64,
    ) -> Result<(), Error> {
        unsafe { API::ffi_set_activation_descriptor(desc, mode, relu_nan_opt, relu_ceiling) }
    }

    /// Computes an activation forward function.
    pub fn activation_forward(
        handle: cudnnHandle_t,
        activation_desc: cudnnActivationDescriptor_t,
        alpha: *const ::libc::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_activation_forward(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y)
        }
    }

    /// Computes an activation backward function.
    pub fn activation_backward(
        handle: cudnnHandle_t,
        activation_desc: cudnnActivationDescriptor_t,
        alpha: *const ::libc::c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *const ::libc::c_void,
        dy_desc: cudnnTensorDescriptor_t,
        dy: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        dx_desc: cudnnTensorDescriptor_t,
        dx: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_activation_backward(
                handle,
                activation_desc,
                alpha,
                y_desc,
                y,
                dy_desc,
                dy,
                beta,
                x_desc,
                x,
                dx_desc,
                dx,
            )
        }
    }

    unsafe fn ffi_activation_forward(
        handle: cudnnHandle_t,
        activation_desc: cudnnActivationDescriptor_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnActivationForward(handle, activation_desc, alpha, src_desc, src_data, beta, dest_desc, dest_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("`mode` is invalid or dimensions of input and output tensor differ or `data_type` or strides of the tensors differ.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute activation forward.")),
        }
    }

    unsafe fn ffi_activation_backward(
        handle: cudnnHandle_t,
        activation_desc: cudnnActivationDescriptor_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        src_diff_desc: cudnnTensorDescriptor_t,
        src_diff_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: cudnnTensorDescriptor_t,
        dest_diff_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnActivationBackward(handle, activation_desc, alpha, src_desc, src_data, src_diff_desc, src_diff_data, dest_desc, dest_data, beta, dest_diff_desc, dest_diff_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("`mode` is invalid or dimensions of input and output tensor differ or `data_type` or strides of the tensors differ.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("`mode` is invalid or dimensions of input and output tensor differ or `data_type` or strides of the tensors differ.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute activation backward.")),
        }
    }

    unsafe fn ffi_create_activation_descriptor() -> Result<cudnnActivationDescriptor_t, Error> {
        let mut desc: cudnnActivationDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateActivationDescriptor(&mut desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {
                Err(Error::AllocFailed("The resources could not be allocated."))
            }
            _ => Err(Error::Unknown(
                "Unable to create generic CUDA cuDNN Activation Descriptor.",
            )),
        }
    }

    unsafe fn ffi_destroy_activation_descriptor(
        desc: cudnnActivationDescriptor_t,
    ) -> Result<(), Error> {
        match cudnnDestroyActivationDescriptor(desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(Error::Unknown(
                "Unable to destroy CUDA cuDNN Activation Descriptor.",
            )),
        }
    }

    unsafe fn ffi_set_activation_descriptor(
        desc: cudnnActivationDescriptor_t,
        mode: cudnnActivationMode_t,
        relu_nan_opt: cudnnNanPropagation_t,
        relu_ceiling: f64,
    ) -> Result<(), Error> {
        match cudnnSetActivationDescriptor(desc, mode, relu_nan_opt, relu_ceiling) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "`window_dim_a`, `padding_a` or `stride_a` has negative element or invalid `mode`.",
            )), // FIXME
            _ => Err(Error::Unknown(
                "Unable to set CUDA cuDNN Activation Descriptor.",
            )),
        }
    }
}
