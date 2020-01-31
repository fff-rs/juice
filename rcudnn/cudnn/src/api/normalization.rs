//! Provides the normalization functionality from the CUDA cuDNN API.
//!
//! This includes divisive normalization and Local Response Normalization.

use crate::ffi::*;
use crate::{Error, API};

impl API {
    /// Creates a generic CUDA cuDNN LRN Descriptor.
    pub fn create_lrn_descriptor() -> Result<cudnnLRNDescriptor_t, Error> {
        unsafe { API::ffi_create_lrn_descriptor() }
    }

    /// Destroys a CUDA cuDNN LRN Descriptor.
    ///
    /// Should be called when freeing a CUDA::Descriptor to not trash up the CUDA device.
    pub fn destroy_lrn_descriptor(desc: cudnnLRNDescriptor_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy_lrn_descriptor(desc) }
    }

    /// Initializes a generic CUDA cuDNN LRN Descriptor with specific properties.
    pub fn set_lrn_descriptor(
        desc: cudnnLRNDescriptor_t,
        lrn_n: ::libc::c_uint,
        lrn_alpha: ::libc::c_double,
        lrn_beta: ::libc::c_double,
        lrn_k: ::libc::c_double,
    ) -> Result<(), Error> {
        unsafe { API::ffi_set_lrn_descriptor(desc, lrn_n, lrn_alpha, lrn_beta, lrn_k) }
    }

    /// Computes an LRN cross channel forward function.
    pub fn lrn_cross_channel_forward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnLRNMode_t,
        alpha: *const ::libc::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_lrn_cross_channel_forward(
                handle, norm_desc, mode, alpha, x_desc, x, beta, y_desc, y,
            )
        }
    }

    /// Computes an LRN cross channel backward function.
    pub fn lrn_cross_channel_backward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnLRNMode_t,
        alpha: *const ::libc::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        dx_desc: cudnnTensorDescriptor_t,
        dx: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *const ::libc::c_void,
        dy_desc: cudnnTensorDescriptor_t,
        dy: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_lrn_cross_channel_backward(
                handle, norm_desc, mode, alpha, x_desc, x, dx_desc, dx, beta, y_desc, y, dy_desc,
                dy,
            )
        }
    }

    /// Computes an devisive normalization forward function.
    pub fn divisive_normalization_forward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnDivNormMode_t,
        alpha: *const ::libc::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        means: *const ::libc::c_void,
        temp: *mut ::libc::c_void,
        temp2: *mut ::libc::c_void,
        beta: *const ::libc::c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_divisive_normalization_forward(
                handle, norm_desc, mode, alpha, x_desc, x, means, temp, temp2, beta, y_desc, y,
            )
        }
    }

    /// Computes an devisive normalization backward function.
    pub fn divisive_normalization_backward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnDivNormMode_t,
        alpha: *const ::libc::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        means: *const ::libc::c_void,
        dy: *const ::libc::c_void,
        temp: *mut ::libc::c_void,
        temp2: *mut ::libc::c_void,
        beta: *const ::libc::c_void,
        dx_dmeans_desc: cudnnTensorDescriptor_t,
        dx: *mut ::libc::c_void,
        dmeans: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_divisive_normalization_backward(
                handle,
                norm_desc,
                mode,
                alpha,
                x_desc,
                x,
                means,
                dy,
                temp,
                temp2,
                beta,
                dx_dmeans_desc,
                dx,
                dmeans,
            )
        }
    }

    unsafe fn ffi_create_lrn_descriptor() -> Result<cudnnLRNDescriptor_t, Error> {
        let mut desc: cudnnLRNDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateLRNDescriptor(&mut desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {
                Err(Error::AllocFailed("The resources could not be allocated."))
            }
            _ => Err(Error::Unknown(
                "Unable to create generic CUDA cuDNN LRN Descriptor.",
            )),
        }
    }

    unsafe fn ffi_destroy_lrn_descriptor(desc: cudnnLRNDescriptor_t) -> Result<(), Error> {
        match cudnnDestroyLRNDescriptor(desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(Error::Unknown(
                "Unable to destroy CUDA cuDNN LRN Descriptor.",
            )),
        }
    }

    unsafe fn ffi_set_lrn_descriptor(
        desc: cudnnLRNDescriptor_t,
        lrn_n: ::libc::c_uint,
        lrn_alpha: ::libc::c_double,
        lrn_beta: ::libc::c_double,
        lrn_k: ::libc::c_double,
    ) -> Result<(), Error> {
        match cudnnSetLRNDescriptor(desc, lrn_n, lrn_alpha, lrn_beta, lrn_k) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "One of the input parameters was out of range.",
            )),
            _ => Err(Error::Unknown(
                "Unable to set CUDA cuDNN Pooling Descriptor.",
            )),
        }
    }

    unsafe fn ffi_lrn_cross_channel_forward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnLRNMode_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnLRNCrossChannelForward(handle, norm_desc, mode, alpha, src_desc, src_data, beta, dest_desc, dest_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the tensor pointers  `src_data`, `dest_data` is NULL. Number of input tensor dimensions is 2 or less. LRN Descriptor params are out of valid range. Input Tensor is 5D but is not NCDHW DHW-packed format.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("`data_type`, `sride`, `dimensions` mismatch or tensor strides are negative.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute LRN cross channel forward.")),
        }
    }

    unsafe fn ffi_lrn_cross_channel_backward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnLRNMode_t,
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
        match cudnnLRNCrossChannelBackward(handle, norm_desc, mode, alpha, src_desc, src_data, src_diff_desc, src_diff_data, dest_desc, dest_data, beta, dest_diff_desc, dest_diff_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the tensor pointers  `src_data`, `dest_data` is NULL. Number of input tensor dimensions is 2 or less. LRN Descriptor params are out of valid range. Input Tensor is 5D but is not NCDHW DHW-packed format.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("`data_type`, `sride`, `dimensions` mismatch or tensor strides are negative.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute LRN cross channel backward.")),
        }
    }

    unsafe fn ffi_divisive_normalization_forward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnDivNormMode_t,
        alpha: *const ::libc::c_void,
        x_desc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        means: *const ::libc::c_void,
        temp: *mut ::libc::c_void,
        temp2: *mut ::libc::c_void,
        beta: *const ::libc::c_void,
        y_desc: cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnDivisiveNormalizationForward(handle, norm_desc, mode, alpha, x_desc, x, means, temp, temp2, beta, y_desc, y) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the tensor pointers  `src_data`, `dest_data`, `tmp_data`, `tmp_data2` is NULL. Number of input tensor or output tensor dimensions is outside of [4,5] range. A mismatch in dimensions between any two of the input or output tensors. For in-place computation (`src_data` == `dest_data`), a mismatch in strides between the input data and output data tensors. Alpha or beta pointer is NULL. LRN descriptor parameters are outside or their valid ranges. Any of the tensor strides are negative.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("stried of the input and output tensors mismatch.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute divisive normalization forward.")),
        }
    }

    unsafe fn ffi_divisive_normalization_backward(
        handle: cudnnHandle_t,
        norm_desc: cudnnLRNDescriptor_t,
        mode: cudnnDivNormMode_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        src_means_data: *const ::libc::c_void,
        src_diff_data: *const ::libc::c_void,
        temp_data: *mut ::libc::c_void,
        temp_data2: *mut ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_data_desc: cudnnTensorDescriptor_t,
        dest_data_diff: *mut ::libc::c_void,
        dest_means_diff: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnDivisiveNormalizationBackward(handle, norm_desc, mode, alpha, src_desc, src_data, src_means_data, src_diff_data, temp_data, temp_data2, beta, dest_data_desc, dest_data_diff, dest_means_diff) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: One of the tensor pointers  `src_data`, `dest_data`, `tmp_data`, `tmp_data2` is NULL. Number of input tensor or output tensor dimensions is outside of [4,5] range. A mismatch in dimensions between any two of the input or output tensors. For in-place computation (`src_data` == `dest_data`), a mismatch in strides between the input data and output data tensors. Alpha or beta pointer is NULL. LRN descriptor parameters are outside or their valid ranges. Any of the tensor strides are negative.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("`mode` is invalid or dimensions of input and output tensor differ or `data_type` or strides of the tensors differ.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute divisive normalization backward.")),
        }
    }
}
