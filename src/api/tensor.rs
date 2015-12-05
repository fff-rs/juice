//! Provides the tensor functionality from the CUDA cuDNN API.

use ::{API, Error};
use super::ffi::*;

impl API {
    /// Creates a generic CUDA cuDNN Tensor Descriptor.
    pub fn create_tensor_descriptor() -> Result<cudnnTensorDescriptor_t, Error> {
        unsafe { API::ffi_create_tensor_descriptor() }
    }

    /// Destroys a CUDA cuDNN Tensor Descriptor.
    ///
    /// Should be called when freeing a CUDA::Descriptor to not trash up the CUDA device.
    pub fn destroy_tensor_descriptor(tensor_desc: cudnnTensorDescriptor_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy_tensor_descriptor(tensor_desc) }
    }

    /// Initializes a generic CUDA cuDNN Tensor Descriptor with specific properties.
    pub fn set_tensor_descriptor(
        tensor_desc: cudnnTensorDescriptor_t,
        data_type: cudnnDataType_t,
        nb_dims: ::libc::c_int,
        dim_a: *const ::libc::c_int,
        stride_a: *const ::libc::c_int,
    ) -> Result<(), Error> {
        unsafe { API::ffi_set_tensor_nd_descriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a) }
    }

    unsafe fn ffi_create_tensor_descriptor() -> Result<cudnnTensorDescriptor_t, Error> {
        let mut tensor_desc: cudnnTensorDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateTensorDescriptor(&mut tensor_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(tensor_desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => Err(Error::AllocFailed("The resources could not be allocated.")),
            _ => Err(Error::Unknown("Unable to create generic CUDA cuDNN Tensor Descriptor.")),
        }
    }

    unsafe fn ffi_destroy_tensor_descriptor(tensor_desc: cudnnTensorDescriptor_t) -> Result<(), Error> {
        match cudnnDestroyTensorDescriptor(tensor_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(Error::Unknown("Unable to destroy CUDA cuDNN Tensor Descriptor context.")),
        }
    }

    unsafe fn ffi_set_tensor_nd_descriptor(
        tensor_desc: cudnnTensorDescriptor_t,
        data_type: cudnnDataType_t,
        nb_dims: ::libc::c_int,
        dim_a: *const ::libc::c_int,
        stride_a: *const ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnSetTensorNdDescriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("`dim_a` invalid due to negative or zero value, or invalid `data_type`.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("`nb_dims` exceeds CUDNN_DIM_MAX or 2 Giga-elements.")),
            _ => Err(Error::Unknown("Unable to set CUDA cuDNN Tensor Descriptor.")),
        }
    }

    unsafe fn ffi_set_tensor_4d_descriptor(
        tensor_desc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        data_type: cudnnDataType_t,
        c: ::libc::c_int,
        n: ::libc::c_int,
        w: ::libc::c_int,
        h: ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnSetTensor4dDescriptor(tensor_desc, format, data_type, c, n, w, h) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("`c`, `n`, `w`, `h` was to negative or zero, or `data_type` or `format` had an invalid enum.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("the total size of the tensor exceeds CUDNN_DIM_MAX or 2 Giga-elements.")),
            _ => Err(Error::Unknown("Unable to set CUDA cuDNN 4D Tensor Descriptor.")),
        }
    }
}
