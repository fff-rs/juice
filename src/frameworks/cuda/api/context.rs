//! Provides the Cuda API with its context functionality.
//!
//! A Collenchyma device can be understood as a synonym to Cuda's context.

use libc;
use super::{API, Error};
use frameworks::cuda::Device;
use frameworks::cuda::Context;
use super::ffi::*;
use std::ptr;

impl API {
    /// Creates a Cuda context.
    ///
    /// An Cuda context can only be created with one device. Contexts are used by the Cuda
    /// runtime for managing objects such as command-queues, memory, program and kernel objects
    /// and for executing kernels on one or more devices specified in the context.
    /// An Cuda context is a synonym to a Collenchyma device.
    pub fn create_context(device: Device) -> Result<CUcontext, Error> {
        unsafe {API::ffi_create_context(device.id_c())}
    }

    /// Removes a created Cuda context from the device.
    ///
    /// Should be called when freeing a Cuda::Context to not trash up the Cuda device.
    pub fn destroy_context(context: &mut Context) -> Result<(), Error> {
        unsafe {API::ffi_destroy_context(context.id_c())}
    }

    unsafe fn ffi_create_context(
        dev: CUdevice,
    ) -> Result<CUcontext, Error> {
        let mut context: CUcontext = ptr::null_mut();
        match cuCtxCreate_v2(&mut context, CU_CTX_SCHED_AUTO, dev) {
            CUresult::CUDA_SUCCESS => Ok(context),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_DEVICE => Err(Error::InvalidValue(format!("Invalid value for `device` provided."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            CUresult::CUDA_ERROR_OUT_OF_MEMORY => Err(Error::OutOfMemory(format!("Device is out of memory."))),
            CUresult::CUDA_ERROR_UNKNOWN => Err(Error::Unknown(format!("An unknown Error occured. Check the CUDA DRIVER API manual for more details."))),
            _ => Err(Error::Unknown(format!("Unable to create Cuda context."))),
        }
    }

    unsafe fn ffi_destroy_context (
        ctx: CUcontext,
    ) -> Result<(), Error> {
        match cuCtxDestroy_v2(ctx) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            _ => Err(Error::Unknown(format!("Unable to destroy Cuda context."))),
        }
    }
}
