//! Provides a Rust wrapper around Cuda's memory.

use super::api::{Driver, DriverFFI, DriverError};
use super::api::{Dnn, DnnFFI, DnnError};
use memory::*;

use std::ptr;

#[derive(Debug)]
/// Defines a Cuda Memory.
pub struct Memory {
    id: isize,
    /// CUDA ID to a potential data descriptor.
    /// Is e.g. used by cuDNN with the Tensor Desciptor.
    descriptor: Option<isize>,
    /// Pointer to host memory that is used for pinned host memory.
    host_ptr: *mut u8,
}

impl Drop for Memory {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        Driver::mem_free(self);
    }
}

#[allow(unused_mut)]
impl Memory {
    /// Initializes a new Cuda memory.
    pub fn new(size: usize) -> Result<Memory, DriverError> {
        Driver::mem_alloc(size as u64)
    }

    /// Initializes a new Cuda memory from its C type.
    pub fn from_c(id: DriverFFI::CUdeviceptr, descriptor: Option<DnnFFI::cudnnTensorDescriptor_t>) -> Memory {
        match descriptor {
            Some(desc) => {
                Memory {
                    id: id as isize,
                    descriptor: Some(desc as isize),
                    host_ptr: ptr::null_mut(),
                }
            },
            None => {
                Memory {
                    id: id as isize,
                    descriptor: None,
                    host_ptr: ptr::null_mut(),
                }
            }
        }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the memory id as its C type.
    pub fn id_c(&self) -> DriverFFI::CUdeviceptr {
        self.id as DriverFFI::CUdeviceptr
    }
}

impl IMemory for Memory {}
