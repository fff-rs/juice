//! Provides a Rust wrapper around Cuda's memory.

use super::api::{Driver, DriverError, DriverFFI};
use crate::device::IMemory;

use std::{fmt, ptr};

/// Defines a Cuda Memory.
pub struct Memory {
    id: DriverFFI::CUdeviceptr,
    /// Pointer to host memory that is used for pinned host memory.
    host_ptr: *mut u8,
}

impl fmt::Debug for Memory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Memory({})", self.id)
    }
}

impl Drop for Memory {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        Driver::mem_free(*self.id_c());
    }
}

#[allow(unused_mut)]
impl Memory {
    /// Initializes a new Cuda memory.
    pub fn new(size: usize) -> Result<Memory, DriverError> {
        Driver::mem_alloc(size)
    }

    /// Initializes a new Cuda memory from its C type.
    pub fn from_c(id: DriverFFI::CUdeviceptr) -> Memory {
        Memory {
            id,
            host_ptr: ptr::null_mut(),
        }
    }

    /// Returns the memory id as its C type.
    pub fn id_c(&self) -> &DriverFFI::CUdeviceptr {
        &self.id
    }
}

impl IMemory for Memory {}
