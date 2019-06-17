//! Defines Cuda Device Memory.
//!
//! Hold a pointer and size of the cuda device memory.
//! This is only a convenience wrapper to interact in a
//! defined mamner with cudnn, which requires scrap/temporary
//! memory for some operations, i.e. dropout.

use super::{Error, API};

#[derive(Debug)]
/// A pointer to memory existing on a nvidia GPU
pub struct CudaDeviceMemory {
    ptr: *mut ::libc::c_void,
    size: usize,
}

impl CudaDeviceMemory {
    /// Saw fun X Y Z
    pub fn new(size: usize) -> Result<CudaDeviceMemory, Error> {
        let ptr = API::cuda_allocate_device_memory(size)?;
        Ok(CudaDeviceMemory {
            ptr: ptr,
            size: size,
        })
    }

    /// Initializes a new CUDA Device Memory from its C type.
    pub fn from_c(ptr: *mut ::libc::c_void, size: usize) -> CudaDeviceMemory {
        CudaDeviceMemory {
            ptr: ptr,
            size: size,
        }
    }

    /// Returns the CUDA Device Memory ptr as its C type.
    pub fn id_c(&self) -> &*mut ::libc::c_void {
        &self.ptr
    }

    /// Returns the size of the CUDA Device Memory chunk.
    pub fn size(&self) -> &usize {
        &self.size
    }
}

impl Drop for CudaDeviceMemory {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        self.size = 0;
        API::cuda_free_device_memory(*self.id_c()).unwrap()
    }
}
