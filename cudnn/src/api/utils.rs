//! Provides utility functionality for the CUDA cuDNN API.

use ::{API, Error};
use ffi::*;
use std::ptr;

impl API {
    /// Initialize the CUDA cuDNN API with needed context and resources.
    ///
    /// The returned `handle` must be provided to future CUDA cuDNN API calls.
    /// Call this method outside of performance critical routines.
    pub fn init() -> Result<cudnnHandle_t, Error> {
        Ok( unsafe { API::ffi_create() }? )
    }

    /// Destroys the CUDA cuDNN context and resources associated with the `handle`.
    ///
    /// Frees up resources and will call `cudaDeviceSynchronize` internaly.
    /// Therefore, use this method outside of performance critical routines.
    pub fn destroy(handle: cudnnHandle_t) -> Result<(), Error> {
        unsafe { API::ffi_destroy(handle) }
    }

    /// Returns the version of the CUDA cuDNN API.
    pub fn get_version() -> usize {
        unsafe { API::ffi_get_version() }
    }

    unsafe fn ffi_get_version() -> ::libc::size_t {
        cudnnGetVersion()
    }

    unsafe fn ffi_create() -> Result<cudnnHandle_t, Error> {
        let mut handle: cudnnHandle_t = ptr::null_mut();
        match cudnnCreate(&mut handle) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(handle),
            cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized("CUDA Driver/Runtime API not initialized.")),
            cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch("cuDNN only supports devices with compute capabilities greater than or equal to 3.0.")),
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => Err(Error::AllocFailed("The resources could not be allocated.")),
            _ => Err(Error::Unknown("Unable to create the CUDA cuDNN context/resources."))
        }
    }

    unsafe fn ffi_destroy(handle: cudnnHandle_t) -> Result<(), Error> {
        match cudnnDestroy(handle) {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => Ok(()),
            cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized("CUDA Driver/Runtime API not initialized.")),
            _ => Err(Error::Unknown("Unable to destroy the CUDA cuDNN context/resources.")),
        }
    }
}
