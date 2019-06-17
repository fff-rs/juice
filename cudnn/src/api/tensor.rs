//! Provides the tensor functionality from the CUDA cuDNN API.
//!
//! This includes the Tensor Descriptor as well as other Tensor functionality,
//! such as transformation and co..

use ffi::*;
use {Error, API};

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
        unsafe {
            API::ffi_set_tensor_nd_descriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a)
        }
    }

    /// Returns informations about a generic CUDA cuDNN Tensor Descriptor.
    pub fn get_tensor_descriptor(
        tensor_desc: cudnnTensorDescriptor_t,
        nb_dims_requested: ::libc::c_int,
        data_type: *mut cudnnDataType_t,
        nb_dims: *mut ::libc::c_int,
        dim_a: *mut ::libc::c_int,
        stride_a: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_get_tensor_nd_descriptor(
                tensor_desc,
                nb_dims_requested,
                data_type,
                nb_dims,
                dim_a,
                stride_a,
            )
        }
    }

    /// Transforms a CUDA cuDNN Tensor from to another Tensor with a different layout.
    ///
    /// This function copies the scaled data from one tensor to another tensor with a different
    /// layout. Those descriptors need to have the same dimensions but not necessarily the
    /// same strides. The input and output tensors must not overlap in any way (i.e., tensors
    /// cannot be transformed in place). This function can be used to convert a tensor with an
    /// unsupported format to a supported one.
    pub fn transform_tensor(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_transform_tensor(
                handle, alpha, src_desc, src_data, beta, dest_desc, dest_data,
            )
        }
    }

    /// Adds the scaled values from one a CUDA cuDNN Tensor to another.
    ///
    /// Up to dimension 5, all tensor formats are supported. Beyond those dimensions, this
    /// routine is not supported.
    ///
    /// This function adds the scaled values of one bias tensor to another tensor. Each dimension
    /// of the `bias` tensor must match the coresponding dimension of the `src_dest` tensor or
    /// must be equal to 1. In the latter case, the same value from the bias tensor for thoses
    /// dimensions will be used to blend into the `src_dest` tensor.
    pub fn add_tensor(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        bias_desc: cudnnTensorDescriptor_t,
        bias_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        src_dest_desc: cudnnTensorDescriptor_t,
        src_dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_add_tensor(
                handle,
                alpha,
                bias_desc,
                bias_data,
                beta,
                src_dest_desc,
                src_dest_data,
            )
        }
    }

    /// Sets all elements of a tensor to a given value.
    pub fn set_tensor(
        handle: cudnnHandle_t,
        src_dest_desc: cudnnTensorDescriptor_t,
        src_dest_data: *mut ::libc::c_void,
        value: *const ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe { API::ffi_set_tensor(handle, src_dest_desc, src_dest_data, value) }
    }

    /// Scales all elements of a tensor by a given factor.
    pub fn scale_tensor(
        handle: cudnnHandle_t,
        src_dest_desc: cudnnTensorDescriptor_t,
        src_dest_data: *mut ::libc::c_void,
        alpha: *const ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe { API::ffi_scale_tensor(handle, src_dest_desc, src_dest_data, alpha) }
    }

    unsafe fn ffi_create_tensor_descriptor() -> Result<cudnnTensorDescriptor_t, Error> {
        let mut tensor_desc: cudnnTensorDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateTensorDescriptor(&mut tensor_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(tensor_desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {
                Err(Error::AllocFailed("The resources could not be allocated."))
            }
            _ => Err(Error::Unknown(
                "Unable to create generic CUDA cuDNN Tensor Descriptor.",
            )),
        }
    }

    unsafe fn ffi_destroy_tensor_descriptor(
        tensor_desc: cudnnTensorDescriptor_t,
    ) -> Result<(), Error> {
        match cudnnDestroyTensorDescriptor(tensor_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(Error::Unknown(
                "Unable to destroy CUDA cuDNN Tensor Descriptor context.",
            )),
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
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "`dim_a` invalid due to negative or zero value, or invalid `data_type`.",
            )),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported(
                "`nb_dims` exceeds CUDNN_DIM_MAX or 2 Giga-elements.",
            )),
            _ => Err(Error::Unknown(
                "Unable to set CUDA cuDNN Tensor Descriptor.",
            )),
        }
    }

    unsafe fn ffi_get_tensor_nd_descriptor(
        tensor_desc: cudnnTensorDescriptor_t,
        nb_dims_requested: ::libc::c_int,
        data_type: *mut cudnnDataType_t,
        nb_dims: *mut ::libc::c_int,
        dim_a: *mut ::libc::c_int,
        stride_a: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnGetTensorNdDescriptor(
            tensor_desc,
            nb_dims_requested,
            data_type,
            nb_dims,
            dim_a,
            stride_a,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "`dim_a` invalid due to negative or zero value, or invalid `data_type`.",
            )),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported(
                "`nb_dims` exceeds CUDNN_DIM_MAX or 2 Giga-elements.",
            )),
            _ => Err(Error::Unknown(
                "Unable to set CUDA cuDNN Tensor Descriptor.",
            )),
        }
    }

    unsafe fn ffi_transform_tensor(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnTransformTensor(handle, alpha, src_desc, src_data, beta, dest_desc, dest_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("the dimensions n, c, h, w or the data type of the two tensor descriptors are different.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to transform CUDA cuDNN Tensor.")),
        }
    }

    unsafe fn ffi_add_tensor(
        handle: cudnnHandle_t,
        alpha: *const ::libc::c_void,
        bias_desc: cudnnTensorDescriptor_t,
        bias_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        src_dest_desc: cudnnTensorDescriptor_t,
        src_dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnAddTensor(handle, alpha, bias_desc, bias_data, beta, src_dest_desc, src_dest_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("The dimensions of the bias tensor refer to an amount of data that is incompatible the output tensor dimensions or the  data type of the two tensor descriptors are different.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("The dimensions of the bias tensor and the output tensor dimensions are above 5.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to add CUDA cuDNN Tensor.")),
        }
    }

    unsafe fn ffi_set_tensor(
        handle: cudnnHandle_t,
        src_dest_desc: cudnnTensorDescriptor_t,
        src_dest_data: *mut ::libc::c_void,
        value: *const ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnSetTensor(handle, src_dest_desc, src_dest_data, value) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                Err(Error::BadParam("One of the provided pointers is NULL."))
            }
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => {
                Err(Error::ExecutionFailed("Execution failed to launch on GPU."))
            }
            _ => Err(Error::Unknown("Unable to set CUDA cuDNN Tensor.")),
        }
    }

    unsafe fn ffi_scale_tensor(
        handle: cudnnHandle_t,
        src_dest_desc: cudnnTensorDescriptor_t,
        src_dest_data: *mut ::libc::c_void,
        alpha: *const ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnScaleTensor(handle, src_dest_desc, src_dest_data, alpha) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {
                Err(Error::BadParam("One of the provided pointers is NULL."))
            }
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => {
                Err(Error::ExecutionFailed("Execution failed to launch on GPU."))
            }
            _ => Err(Error::Unknown("Unable to scale CUDA cuDNN Tensor.")),
        }
    }
}
