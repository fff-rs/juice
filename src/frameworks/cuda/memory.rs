//! Provides a Rust wrapper around Cuda's memory.

use super::api::ffi::*;
use super::api::{API, Error};
use memory::*;

use std::ptr;

#[derive(Debug)]
/// Defines a Cuda Memory.
pub struct Memory {
    id: isize,
    /// Pointer to host memory that is used for pinned host memory.
    host_ptr: *mut u8,
}

impl Drop for Memory {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::mem_free(self);
    }
}

#[allow(unused_mut)]
impl Memory {
    /// Initializes a new Cuda memory.
    pub fn new(size: usize) -> Result<Memory, Error> {
        API::mem_alloc(size as u64)
    }

    /// Initializes a new Cuda memory from its C type.
    pub fn from_c(id: CUdeviceptr) -> Memory {
        Memory {
            id: id as isize,
            host_ptr: ptr::null_mut(),
        }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the memory id as its C type.
    pub fn id_c(&self) -> CUdeviceptr {
        self.id as CUdeviceptr
    }
}

impl IMemory for Memory {}
