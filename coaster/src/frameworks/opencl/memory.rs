#![allow(missing_docs)]
use super::api::types as cl;
use super::api::{API, Error};
use super::Context;
use device::IMemory;

use std::{ptr, fmt};

/// Holds a OpenCL memory id and manages its deallocation
pub struct Memory {
    /// The underlying memory id>
    memory: cl::memory_id,
    memory_flags: MemoryFlags,

    /// Pointer to host memory that is used for pinned host memory.
    host_ptr: *mut u8,
}

impl fmt::Debug for Memory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Memory({:?}, {:?})", self.memory, self.memory_flags)
    }
}

impl Drop for Memory {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::release_memory(self);
        if self.memory_flags.contains(MemoryFlags::MEM_USE_HOST_PTR) {
            unsafe {
                Box::from_raw(self.host_ptr);
            }
        }
    }
}

bitflags! {
    pub struct MemoryFlags: cl::bitfield {
        const MEM_READ_WRITE       = 1 << 0;
        const MEM_WRITE_ONLY       = 1 << 1;
        const MEM_READ_ONLY        = 1 << 2;
        const MEM_USE_HOST_PTR     = 1 << 3;
        const MEM_ALLOC_HOST_PTR   = 1 << 4;
        const MEM_COPY_HOST_PTR    = 1 << 5;
    }
}

impl Default for MemoryFlags {
    fn default() -> MemoryFlags {
        MemoryFlags::MEM_READ_WRITE
    }
}

#[allow(unused_mut)]
impl Memory {
    pub fn new(context: &Context, size: usize) -> Result<Memory, Error> {
        API::create_buffer(context, MemoryFlags::default(), size, None)
    }

    pub fn id_c(&self) -> cl::memory_id {
        self.memory
    }

    pub fn from_c(id: cl::memory_id) -> Memory {
        Memory {
            memory: id,
            memory_flags: MemoryFlags::default(),
            host_ptr: ptr::null_mut(),
        }
    }
}

impl IMemory for Memory {}
