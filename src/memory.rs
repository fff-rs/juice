//! Provides a representation for memory across different frameworks.
//!
//! Memory is allocated by a device in a way that it is accessible for its computations.
//!
//! Normally you will want to use [SharedMemory][shared_mem] which handles synchronization
//! of the latest memory copy to the required device.
//!
//! [shared_mem]: ../shared_mem/index.html

use frameworks::native::flatbox::FlatBox;
use frameworks::opencl::memory::Memory;

/// Specifies Memory behavior accross frameworks.
pub trait IMemory { }

#[derive(Debug)]
/// Container for all known IMemory implementations
pub enum MemoryType {
    /// A OpenCL Context
    Native(FlatBox),
    /// A OpenCL Context
    OpenCL(Memory),
}

impl MemoryType {
    /// Extract the FlatBox if MemoryType is Native.
    pub fn as_native(&self) -> Option<&FlatBox> {
        match *self {
            MemoryType::Native(ref ret) => Some(ret),
            _ => None,
        }
    }

    /// Extract the FlatBox mutably if MemoryType is Native.
    pub fn as_mut_native(&mut self) -> Option<&mut FlatBox> {
        match *self {
            MemoryType::Native(ref mut ret) => Some(ret),
            _ => None,
        }
    }

    /// Extract the OpenCL Memory if MemoryType is OpenCL.
    pub fn as_opencl(&self) -> Option<&Memory> {
        match *self {
            MemoryType::OpenCL(ref ret) => Some(ret),
            _ => None,
        }
    }

    /// Extract the OpenCL Memory mutably if MemoryType is OpenCL.
    pub fn as_mut_opencl(&mut self) -> Option<&mut Memory> {
        match *self {
            MemoryType::OpenCL(ref mut ret) => Some(ret),
            _ => None,
        }
    }
}
