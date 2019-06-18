//! Provides the Cuda API with its memory/buffer functionality.

use super::{API, Error};
use frameworks::native::flatbox::FlatBox;
use frameworks::cuda::Memory;
use super::ffi::*;

impl API {
    /// Allocates memory on the Cuda device.
    ///
    /// Allocates bytesize bytes of linear memory on the device. The allocated memory is suitably
    /// aligned for any kind of variable. The memory is not cleared.
    /// Returns a memory id for the created buffer, which can now be writen to.
    pub fn mem_alloc(bytesize: size_t) -> Result<Memory, Error> {
        Ok(Memory::from_c(unsafe {API::ffi_mem_alloc(bytesize)}?))
    }

    /// Releases allocated memory from the Cuda device.
    pub fn mem_free(memory: CUdeviceptr) -> Result<(), Error> {
        unsafe {API::ffi_mem_free(memory)}
    }

    /// Copies memory from the Host to the Cuda device.
    pub fn mem_cpy_h_to_d(host_mem: &FlatBox, device_mem: &mut Memory) -> Result<(), Error> {
        unsafe {API::ffi_mem_cpy_h_to_d(*device_mem.id_c(), host_mem.as_slice().as_ptr(), host_mem.byte_size())}
    }

    /// Copies memory from the Cuda device to the Host.
    pub fn mem_cpy_d_to_h(device_mem: &Memory, host_mem: &mut FlatBox) -> Result<(), Error> {
        unsafe {API::ffi_mem_cpy_d_to_h(host_mem.as_mut_slice().as_mut_ptr(), *device_mem.id_c(), host_mem.byte_size())}
    }

    unsafe fn ffi_mem_alloc(bytesize: size_t) -> Result<CUdeviceptr, Error> {
        let mut memory_id: CUdeviceptr = 0;
        match cuMemAlloc_v2(&mut memory_id, bytesize) {
            CUresult::CUDA_SUCCESS => Ok(memory_id),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized("CUDA got deinitialized.")),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized("CUDA is not initialized.")),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext("No valid context available.")),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue("Invalid value provided.")),
            CUresult::CUDA_ERROR_OUT_OF_MEMORY => Err(Error::OutOfMemory("Device is out of memory.")),
            _ => Err(Error::Unknown("Unable to allocate memory.")),
        }
    }

    unsafe fn ffi_mem_free(dptr: CUdeviceptr) -> Result<(), Error> {
        match cuMemFree_v2(dptr) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized("CUDA got deinitialized.")),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized("CUDA is not initialized.")),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext("No valid context available.")),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue("Invalid value provided.")),
            _ => Err(Error::Unknown("Unable to free memory.")),
        }
    }

    unsafe fn ffi_mem_cpy_h_to_d(
        dst_device: CUdeviceptr,
        src_host: *const ::libc::c_void,
        byte_count: size_t,
    ) -> Result<(), Error> {
        match cuMemcpyHtoD_v2(dst_device, src_host, byte_count) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized("CUDA got deinitialized.")),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized("CUDA is not initialized.")),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext("No valid context available.")),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue("Invalid value provided.")),
            _ => Err(Error::Unknown("Unable to copy memory from host to device.")),
        }
    }

    unsafe fn ffi_mem_cpy_d_to_h(
        dst_host: *mut ::libc::c_void,
        src_device: CUdeviceptr,
        byte_count: size_t,
    ) -> Result<(), Error> {
        let status = cuMemcpyDtoH_v2(dst_host, src_device, byte_count);
        match status {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized("CUDA got deinitialized.")),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized("CUDA is not initialized.")),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext("No valid context available.")),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue("Invalid value provided.")),
            _ => { println!("{:?}", status); Err(Error::Unknown("Unable to copy memory from device to host.")) },
        }
    }
}
