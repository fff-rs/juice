//! Provides the Cuda API with its memory/buffer functionality.

use libc;
use super::{API, Error};
use frameworks::cuda::{Context, Memory};
use super::types as cl;
use super::ffi::*;

impl API {
    /// Allocates memory on the Cuda device.
    ///
    /// Allocates bytesize bytes of linear memory on the device. The allocated memory is suitably
    /// aligned for any kind of variable. The memory is not cleared.
    /// Returns a memory id for the created buffer, which can now be writen to.
    pub fn mem_alloc(bytesize: size_t) -> Result<CUdeviceptr, Error> {
        let r = unsafe {API::ffi_mem_alloc(bytesize)};
        r
    }

    /// Releases allocated memory from the Cuda device.
    pub fn mem_free(memory: &mut Memory) -> Result<(), Error> {
        unsafe {API::ffi_mem_free(memory.id_c())}
    }

    /// Reads from a buffer to the host memory.
    ///
    /// With write_to_buffer you can do the opposite, write from the host memory to a buffer.
    pub fn read_from_buffer<T>(
        mem: Memory,
        blocking_read: cl::boolean,
        offset: libc::size_t,
        size: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> Result<(), Error> {
        unimplemented!();
    }

    unsafe fn ffi_mem_alloc(bytesize: size_t) -> Result<CUdeviceptr, Error> {
        let mut memory_id: CUdeviceptr = 0;
        match cuMemAlloc_v2(&mut memory_id, bytesize) {
            CUresult::CUDA_SUCCESS => Ok(memory_id),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            CUresult::CUDA_ERROR_OUT_OF_MEMORY => Err(Error::OutOfMemory(format!("Device is out of memory."))),
            _ => Err(Error::Unknown(format!("Unable to allocate memory."))),
        }
    }

    unsafe fn ffi_mem_free(dptr: CUdeviceptr) -> Result<(), Error> {
        match cuMemFree_v2(dptr) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            _ => Err(Error::Unknown(format!("Unable to free memory."))),
        }
    }

    unsafe fn ffi_enqueue_read_buffer(
        command_queue: cl::queue_id,
        buffer: cl::memory_id,
        blocking_read: cl::boolean,
        offset: libc::size_t,
        cb: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event
    ) -> Result<(), Error> {
        unimplemented!()
    }
}
