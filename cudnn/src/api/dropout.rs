//! Provides the dropout functionality from the CUDA cuDNN API.
//!
//! Includes the convolution and filter functionality.

use crate::ffi::*;
use crate::{Error, API};

impl API {
    //
    // cuDNN Dropout
    //

    /// Create a generic CUDA cuDNN DropoutDescriptor
    pub fn create_dropout_descriptor() -> Result<cudnnDropoutDescriptor_t, Error> {
        unsafe { API::ffi_create_dropout_descriptor() }
    }
    /// Destroys a CUDA cuDNN Dropout Descriptor.
    ///
    /// Should be called when freeing a CUDA::Descriptor to not trash up the CUDA device.
    pub fn destroy_dropout_descriptor(dropout_desc: cudnnDropoutDescriptor_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy_dropout_descriptor(dropout_desc) }
    }
    /// Get the states size (GPU memory).
    pub fn dropout_get_states_size(handle: cudnnHandle_t) -> Result<usize, Error> {
        unsafe { API::ffi_dropout_get_states_size(handle) }
    }
    /// Get the reserve space size.
    pub fn dropout_get_reserve_space_size(xdesc: cudnnTensorDescriptor_t) -> Result<usize, Error> {
        unsafe { API::ffi_dropout_get_reserve_space_size(xdesc) }
    }

    /// Initializes a generic CUDA cuDNN Activation Descriptor with specific properties.
    pub fn set_dropout_descriptor(
        dropout_desc: cudnnDropoutDescriptor_t,
        handle: cudnnHandle_t,
        dropout: f32,
        states: *mut ::libc::c_void,
        state_size_in_bytes: usize,
        seed: ::libc::c_ulonglong,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_set_dropout_descriptor(
                dropout_desc,
                handle,
                dropout,
                states,
                state_size_in_bytes,
                seed,
            )
        }
    }

    /// Computes the dropout forward function.
    pub fn dropout_forward(
        handle: cudnnHandle_t,
        dropout_desc: cudnnDropoutDescriptor_t,
        xdesc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        ydesc: cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
        reserve_space: *mut ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_dropout_forward(
                handle,
                dropout_desc,
                xdesc,
                x,
                ydesc,
                y,
                reserve_space,
                reserve_space_size_in_bytes,
            )
        }
    }

    /// Computes the dropout backward function.
    pub fn dropout_backward(
        handle: cudnnHandle_t,
        dropout_desc: cudnnDropoutDescriptor_t,
        dydesc: cudnnTensorDescriptor_t,
        dy: *const ::libc::c_void,
        dxdesc: cudnnTensorDescriptor_t,
        dx: *mut ::libc::c_void,
        reserve_space: *mut ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        unsafe {
            API::ffi_dropout_backward(
                handle,
                dropout_desc,
                dydesc,
                dy,
                dxdesc,
                dx,
                reserve_space,
                reserve_space_size_in_bytes,
            )
        }
    }

    unsafe fn ffi_create_dropout_descriptor() -> Result<cudnnDropoutDescriptor_t, Error> {
        let mut dropout_desc: cudnnDropoutDescriptor_t = ::std::ptr::null_mut();
        match cudnnCreateDropoutDescriptor(&mut dropout_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(dropout_desc),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {
                Err(Error::AllocFailed("The resources could not be allocated"))
            }
            _ => Err(Error::Unknown(
                "Unable create generic CUDA cuDNN Dropout Descriptor",
            )),
        }
    }
    unsafe fn ffi_destroy_dropout_descriptor(
        dropout_desc: cudnnDropoutDescriptor_t,
    ) -> Result<(), Error> {
        match cudnnDestroyDropoutDescriptor(dropout_desc) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            _ => Err(Error::Unknown(
                "Unable to destroy CUDA cuDNN Dropout Descriptor",
            )),
        }
    }
    unsafe fn ffi_dropout_get_states_size(handle: cudnnHandle_t) -> Result<usize, Error> {
        let mut size_in_bytes: usize = 0;
        match cudnnDropoutGetStatesSize(handle, &mut size_in_bytes) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(size_in_bytes),
            _ => Err(Error::Unknown(
                "Unable to get CUDA cuDNN Dropout Descriptor states size",
            )),
        }
    }
    unsafe fn ffi_dropout_get_reserve_space_size(
        xdesc: cudnnTensorDescriptor_t,
    ) -> Result<usize, Error> {
        let mut size_in_bytes: usize = 0;
        match cudnnDropoutGetReserveSpaceSize(xdesc, &mut size_in_bytes) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(size_in_bytes),
            _ => Err(Error::Unknown(
                "Unable to get CUDA cuDNN Dropout Descriptor reserved space size",
            )),
        }
    }
    unsafe fn ffi_set_dropout_descriptor(
        dropout_desc: cudnnDropoutDescriptor_t,
        handle: cudnnHandle_t,
        dropout: f32,
        states: *mut ::libc::c_void,
        state_size_in_bytes: usize,
        seed: ::libc::c_ulonglong,
    ) -> Result<(), Error> {
        match cudnnSetDropoutDescriptor(
            dropout_desc,
            handle,
            dropout,
            states,
            state_size_in_bytes,
            seed,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => Err(Error::InvalidValue(
                "sizeInBytes is less than the value returned by cudnnDropoutGetStatesSize .",
            )),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed(
                "The function failed to launch on the GPU",
            )),
            _ => Err(Error::Unknown(
                "Unable to set CUDA cuDNN Dropout Descriptor",
            )),
        }
    }

    unsafe fn ffi_dropout_forward(
        handle: cudnnHandle_t,
        dropout_desc: cudnnDropoutDescriptor_t,
        xdesc: cudnnTensorDescriptor_t,
        x: *const ::libc::c_void,
        ydesc: cudnnTensorDescriptor_t,
        y: *mut ::libc::c_void,
        reserve_space: *mut ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        match cudnnDropoutForward(
            handle,
            dropout_desc,
            xdesc,
            x,
            ydesc,
            y,
            reserve_space,
            reserve_space_size_in_bytes,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported(
                "The function does not support the provided configuration.",
            )),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "At least one of the following conditions are met: The number of elements of input tensor and output tensors differ, or the datatype of the input tensor and output tensors differs, or the strides of the input tensor and output tensors differ and in-place operation is used (i.e., x and y pointers are equal), or the provided reserveSpaceSizeInBytes is less then the value returned by cudnnDropoutGetReserveSpaceSize, or cudnnSetdropoutDescriptor has not been called on dropoutDesc with the non-NULL states argument",
            )),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed(
                "The function failed to launch on the GPU.",
            )),
            _ => Err(Error::Unknown("Unable to calculate CUDA cuDNN Dropout forward")),

        }
    }
    unsafe fn ffi_dropout_backward(
        handle: cudnnHandle_t,
        dropout_desc: cudnnDropoutDescriptor_t,
        dydesc: cudnnTensorDescriptor_t,
        dy: *const ::libc::c_void,
        dxdesc: cudnnTensorDescriptor_t,
        dx: *mut ::libc::c_void,
        reserve_space: *mut ::libc::c_void,
        reserve_space_size_in_bytes: usize,
    ) -> Result<(), Error> {
        match cudnnDropoutBackward(
            handle,
            dropout_desc,
            dydesc,
            dy,
            dxdesc,
            dx,
            reserve_space,
            reserve_space_size_in_bytes,
        ) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => Err(Error::NotSupported(
                "The function does not support the provided configuration.",
            )),
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => Err(Error::BadParam(
                "At least one of the following conditions are met: The number of elements of input tensor and output tensors differ, or the datatype of the input tensor and output tensors differs, or the strides of the input tensor and output tensors differ and in-place operation is used (i.e., x and y pointers are equal), or the provided reserveSpaceSizeInBytes is less then the value returned by cudnnDropoutGetReserveSpaceSize, or cudnnSetdropoutDescriptor has not been called on dropoutDesc with the non-NULL states argument",
            )),
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed(
                "The function failed to launch on the GPU.",
            )),
            _ => Err(Error::Unknown("Unable to calculate CUDA cuDNN Dropout backward")),
        }
    }
}
