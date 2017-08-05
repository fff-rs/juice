use super::{API, Error};

#[derive(Debug)]
pub struct CudaDeviceMemory {
	ptr: *mut ::libc::c_void,
	size: usize,
}

impl CudaDeviceMemory {
    /// Saw fun X Y Z
    pub fn new(size : usize) -> Result<CudaDeviceMemory,Error> {
        let ptr = API::cuda_allocate_device_memory(size)?;
        Ok(CudaDeviceMemory {
	        ptr: ptr,
	        size: size,
	    })
    }

    /// Initializes a new CUDA cuDNN Tensor Descriptor from its C type.
    pub fn from_c(ptr: *mut ::libc::c_void, size: usize) -> CudaDeviceMemory {
        CudaDeviceMemory { ptr: ptr, size: size }
    }

    /// Returns the CUDA cuDNN Tensor Descriptor as its C type.
    pub fn id_c(&self) -> &*mut ::libc::c_void {
        &self.ptr
    }

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
