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
