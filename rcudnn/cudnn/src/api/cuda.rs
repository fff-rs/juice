//! Provides utility functionality for the CUDA cuDNN API.

use crate::ffi::*;
use crate::{Error, API};
use std::ptr;

impl API {
    /// Initialize the CUDA cuDNN API with needed context and resources.
    ///
    /// The returned `handle` must be provided to future CUDA cuDNN API calls.
    /// Call this method outside of performance critical routines.
    pub fn cuda_allocate_device_memory(bytes: usize) -> Result<*mut ::libc::c_void, Error> {
        unsafe { API::ffi_cuda_allocate_device_memory(bytes) }
    }

    /// Destroys the CUDA cuDNN context and resources associated with the `handle`.
    ///
    /// Frees up resources and will call `cudaDeviceSynchronize` internaly.
    /// Therefore, use this method outside of performance critical routines.
    pub fn cuda_free_device_memory(ptr: *mut ::libc::c_void) -> Result<(), Error> {
        unsafe { API::ffi_cuda_free_device_memory(ptr) }
    }

    unsafe fn ffi_cuda_allocate_device_memory(bytes: usize) -> Result<*mut ::libc::c_void, Error> {
        let mut ptr: *mut ::libc::c_void = ptr::null_mut();
        match cudaMalloc(&mut ptr, bytes) {
            cudaError_t::cudaSuccess => Ok(ptr),
            cudaError_t::cudaErrorMemoryAllocation => {
                Err(Error::AllocFailed("Unable to allocate CUDA device memory."))
            }
            status => Err(Error::Unknown(
                "Unable to allocate CUDA device memory for unknown reasons.",
                status as i32 as u64,
            )),
        }
    }

    unsafe fn ffi_cuda_free_device_memory(ptr: *mut ::libc::c_void) -> Result<(), Error> {
        match cudaFree(ptr) {
            cudaError_t::cudaSuccess => Ok(()),
            // TODO, more error enums sigh
            cudaError_t::cudaErrorInvalidDevicePointer => Err(Error::InvalidValue(
                "Unable to free the CUDA device memory due to invalid device pointer.",
            )),
            cudaError_t::cudaErrorInitializationError => Err(Error::NotInitialized(
                "CUDA Driver/Runtime API not initialized.",
            )),
            status => Err(Error::Unknown(
                "Unable to free the CUDA device memory.",
                status as i32 as u64,
            )),
        }
    }
}
