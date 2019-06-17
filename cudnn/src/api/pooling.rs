//! Provides the pooling functionality from the CUDA cuDNN API.
//!
//! This includes the Pooling Descriptor as well as the Pooling for- and backwar computation.

use ffi::*;
use {Error, API};

impl API {
    /// Creates a generic CUDA cuDNN Pooling Descriptor.
    pub fn create_pooling_descriptor() -> Result<cudnnPoolingDescriptor_t, Error> {
        unsafe { API::ffi_create_pooling_descriptor() }
    }

    /// Destroys a CUDA cuDNN Pooling Descriptor.
    ///
    /// Should be called when freeing a CUDA::Descriptor to not trash up the CUDA device.
    pub fn destroy_pooling_descriptor(desc: cudnnPoolingDescriptor_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy_pooling_descriptor(desc) }
    }

    /// Initializes a generic CUDA cuDNN Pooling Descriptor with specific properties.
    pub fn set_pooling_descriptor(
        desc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        maxpooling_nan_opt: cudnnNanPropagation_t,
        nb_dims: ::libc::c_int,
        window: *const ::libc::c_int,
        padding: *const ::libc::c_int,
        stride: *const ::libc::c_int,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_set_pooling_nd_descriptor(
                desc,
                mode,
                maxpooling_nan_opt,
                nb_dims,
                window,
                padding,
                stride,
            )
        }
    }

    /// Return information about a generic CUDA cuDNN Pooling Descriptor.
    pub fn get_pooling_descriptor(
        desc: cudnnPoolingDescriptor_t,
        nb_dims_requested: ::libc::c_int,
        mode: *mut cudnnPoolingMode_t,
        maxpooling_nan_opt: *mut cudnnNanPropagation_t,
        nb_dims: *mut ::libc::c_int,
        window: *mut ::libc::c_int,
        padding: *mut ::libc::c_int,
        stride: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_get_pooling_nd_descriptor(
                desc,
                nb_dims_requested,
                mode,
                maxpooling_nan_opt,
                nb_dims,
                window,
                padding,
                stride,
            )
        }
    }

    /// Initializes a generic CUDA cuDNN Pooling Descriptor with specific properties.
    pub fn set_pooling_2d_descriptor(
        desc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        nan_propagation: cudnnNanPropagation_t,
        window_height: ::libc::c_int,
        window_width: ::libc::c_int,
        vertical_padding: ::libc::c_int,
        horizontal_padding: ::libc::c_int,
        vertical_stride: ::libc::c_int,
        horizontal_stride: ::libc::c_int,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_set_pooling_2d_descriptor(
                desc,
                mode,
                nan_propagation,
                window_height,
                window_width,
                vertical_padding,
                horizontal_padding,
                vertical_stride,
                horizontal_stride,
            )
        }
    }

    /// Return information about a generic CUDA cuDNN Pooling Descriptor.
    pub fn get_pooling_2d_descriptor(
        desc: cudnnPoolingDescriptor_t,
        mode: *mut cudnnPoolingMode_t,
        nan_propagation: *mut cudnnNanPropagation_t,
        window_height: *mut ::libc::c_int,
        window_width: *mut ::libc::c_int,
        vertical_padding: *mut ::libc::c_int,
        horizontal_padding: *mut ::libc::c_int,
        vertical_stride: *mut ::libc::c_int,
        horizontal_stride: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_get_pooling_2d_descriptor(
                desc,
                mode,
                nan_propagation,
                window_height,
                window_width,
                vertical_padding,
                horizontal_padding,
                vertical_stride,
                horizontal_stride,
            )
        }
    }

    /// Initializes a generic CUDA cuDNN Pooling Descriptor with specific properties.
    pub fn get_pooling_forward_output_dim(
        pooling_desc: cudnnPoolingDescriptor_t,
        input_desc: cudnnTensorDescriptor_t,
        nb_dims: ::libc::c_int,
        out_dim_a: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_get_pooling_nd_forward_output_dim(pooling_desc, input_desc, nb_dims, out_dim_a)
        }
    }

    /// Computes a pooling forward function.
    pub fn pooling_forward(
        handle: cudnnHandle_t,
        pooling_desc: cudnnPoolingDescriptor_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_pooling_forward(
                handle,
                pooling_desc,
                alpha,
                src_desc,
                src_data,
                beta,
                dest_desc,
                dest_data,
            )
        }
    }

    /// Computes a pooling backward function.
    pub fn pooling_backward(
        handle: cudnnHandle_t,
        pooling_desc: cudnnPoolingDescriptor_t,
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
        unsafe {
            API::ffi_pooling_backward(
                handle,
                pooling_desc,
                alpha,
                src_desc,
                src_data,
                src_diff_desc,
                src_diff_data,
                beta,
                dest_desc,
                dest_data,
                dest_diff_desc,
                dest_diff_data,
            )
        }
    }

    unsafe fn ffi_create_pooling_descriptor() -> Result<cudnnPoolingDescriptor_t, Error> {
        let mut desc: cudnnPoolingDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreatePoolingDescriptor(&mut desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {
                Err(Error::AllocFailed("The resources could not be allocated."))
            }
            _ => Err(Error::Unknown(
                "Unable to create generic CUDA cuDNN Pooling Descriptor.",
            )),
        }
    }

    unsafe fn ffi_destroy_pooling_descriptor(desc: cudnnPoolingDescriptor_t) -> Result<(), Error> {
        match cudnnDestroyPoolingDescriptor(desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(Error::Unknown(
                "Unable to destroy CUDA cuDNN Pooling Descriptor.",
            )),
        }
    }

    unsafe fn ffi_set_pooling_nd_descriptor(
        desc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        maxpooling_nan_opt: cudnnNanPropagation_t,
        nb_dims: ::libc::c_int,
        window_dim_a: *const ::libc::c_int,
        padding_a: *const ::libc::c_int,
        stride_a: *const ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnSetPoolingNdDescriptor(
            desc,
            mode,
            maxpooling_nan_opt,
            nb_dims,
            window_dim_a,
            padding_a,
            stride_a,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "`window_dim_a`, `padding_a` or `stride_a` has negative element or invalid `mode`.",
            )),
            _ => Err(Error::Unknown(
                "Unable to set CUDA cuDNN Pooling Descriptor.",
            )),
        }
    }

    unsafe fn ffi_get_pooling_nd_descriptor(
        desc: cudnnPoolingDescriptor_t,
        nb_dims_requested: ::libc::c_int,
        mode: *mut cudnnPoolingMode_t,
        maxpooling_nan_opt: *mut cudnnNanPropagation_t,
        nb_dims: *mut ::libc::c_int,
        window_dim_a: *mut ::libc::c_int,
        padding_a: *mut ::libc::c_int,
        stride_a: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnGetPoolingNdDescriptor(
            desc,
            nb_dims_requested,
            mode,
            maxpooling_nan_opt,
            nb_dims,
            window_dim_a,
            padding_a,
            stride_a,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "`window_dim_a`, `padding_a` or `stride_a` has negative element or invalid `mode`.",
            )),
            _ => Err(Error::Unknown(
                "Unable to get CUDA cuDNN Pooling Descriptor.",
            )),
        }
    }

    unsafe fn ffi_set_pooling_2d_descriptor(
        desc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        maxpooling_nan_opt: cudnnNanPropagation_t,
        window_height: ::libc::c_int,
        window_width: ::libc::c_int,
        vertical_padding: ::libc::c_int,
        horizontal_padding: ::libc::c_int,
        vertical_stride: ::libc::c_int,
        horizontal_stride: ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnSetPooling2dDescriptor(
            desc,
            mode,
            maxpooling_nan_opt,
            window_height,
            window_width,
            vertical_padding,
            horizontal_padding,
            vertical_stride,
            horizontal_stride,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "`window_dim_a`, `padding_a` or `stride_a` has negative element or invalid `mode`.",
            )),
            _ => Err(Error::Unknown(
                "Unable to set CUDA cuDNN Pooling Descriptor 2D.",
            )),
        }
    }

    unsafe fn ffi_get_pooling_2d_descriptor(
        desc: cudnnPoolingDescriptor_t,
        mode: *mut cudnnPoolingMode_t,
        maxpooling_nan_opt: *mut cudnnNanPropagation_t,
        window_height: *mut ::libc::c_int,
        window_width: *mut ::libc::c_int,
        vertical_padding: *mut ::libc::c_int,
        horizontal_padding: *mut ::libc::c_int,
        vertical_stride: *mut ::libc::c_int,
        horizontal_stride: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnGetPooling2dDescriptor(
            desc,
            mode,
            maxpooling_nan_opt,
            window_height,
            window_width,
            vertical_padding,
            horizontal_padding,
            vertical_stride,
            horizontal_stride,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "`window_dim_a`, `padding_a` or `stride_a` has negative element or invalid `mode`.",
            )),
            _ => Err(Error::Unknown(
                "Unable to get CUDA cuDNN Pooling Descriptor 2D.",
            )),
        }
    }

    unsafe fn ffi_get_pooling_nd_forward_output_dim(
        pooling_desc: cudnnPoolingDescriptor_t,
        input_desc: cudnnTensorDescriptor_t,
        nb_dims: ::libc::c_int,
        out_dim_a: *mut ::libc::c_int,
    ) -> Result<(), Error> {
        match cudnnGetPoolingNdForwardOutputDim(pooling_desc, input_desc, nb_dims, out_dim_a) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("`pooling_desc` not initialized or `nb_dims` is inconsistent with `pooling_desc` and `input_desc`.")),
            _ => Err(Error::Unknown("Unable to get CUDA cuDNN Pooling Forward Output dimensions.")),
        }
    }

    unsafe fn ffi_pooling_forward(
        handle: cudnnHandle_t,
        desc: cudnnPoolingDescriptor_t,
        alpha: *const ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        src_data: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        dest_desc: cudnnTensorDescriptor_t,
        dest_data: *mut ::libc::c_void,
    ) -> Result<(), Error> {
        match cudnnPoolingForward(handle, desc, alpha, src_desc, src_data, beta, dest_desc, dest_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: The dimensions n, c of the input tensor and output tensors differ. The datatype of the input tensor and output tensors differs.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("The `w_stride` of input tensor or output tensor is not 1.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute pooling forward.")),
        }
    }

    unsafe fn ffi_pooling_backward(
        handle: cudnnHandle_t,
        desc: cudnnPoolingDescriptor_t,
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
        match cudnnPoolingBackward(handle, desc, alpha, src_desc, src_data, src_diff_desc, src_diff_data, dest_desc, dest_data, beta, dest_diff_desc, dest_diff_data) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam("At least one of the following conditions are met: The dimensions n,c,h,w of the `src_desc` and `src_diff_desc` tensors differ. The strides nStride, cStride, hStride, wStride of the `src_desc` and `src_diff_desc` tensors differ. The dimensions n,c,h,w of the `dest_desc` and `dest_diff_desc` tensors differ. The strides nStride, cStride, hStride, wStride of the `dest_desc` and `dest_diff_desc` tensors differ. The datatype of the four tensors differ.")),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported("The `w_stride` of input tensor or output tensor is not 1.")),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed("Execution failed to launch on GPU.")),
            _ => Err(Error::Unknown("Unable to compute pooling backward.")),
        }
    }
}
