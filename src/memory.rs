//! Provides a representation for memory across different frameworks.
//!
//! Memory is allocated by a device in a way that it is accessible for its computations.
//!
//! Normally you will want to use [SharedMemory][shared_mem] which handles synchronization
//! of the latest memory copy to the required device.
//!
//! [shared_mem]: ../shared_mem/index.html

use frameworks::native::flatbox::FlatBox;
use frameworks::opencl::memory::Memory as OpenCLMemory;
#[cfg(feature = "cuda")]
use frameworks::cuda::memory::Memory as CudaMemory;

/// Specifies Memory behavior accross frameworks.
pub trait IMemory { }

#[derive(Debug)]
/// Container for all known IMemory implementations
pub enum MemoryType {
    /// A Native Memory representation.
    Native(FlatBox),
    /// A OpenCL Memory representation.
    OpenCL(OpenCLMemory),
    /// A Cuda Memory representation.
    #[cfg(feature = "cuda")]
    Cuda(CudaMemory),
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
    pub fn as_opencl(&self) -> Option<&OpenCLMemory> {
        match *self {
            MemoryType::OpenCL(ref ret) => Some(ret),
            _ => None,
        }
    }

    /// Extract the OpenCL Memory mutably if MemoryType is OpenCL.
    pub fn as_mut_opencl(&mut self) -> Option<&mut OpenCLMemory> {
        match *self {
            MemoryType::OpenCL(ref mut ret) => Some(ret),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    /// Extract the Cuda Memory if MemoryType is Cuda
    pub fn as_cuda(&self) -> Option<&CudaMemory> {
        match *self {
            MemoryType::Cuda(ref ret) => Some(ret),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    /// Extract the Cuda Memory mutably if MemoryType is Cuda
    pub fn as_mut_cuda(&mut self) -> Option<&mut CudaMemory> {
        match *self {
            MemoryType::Cuda(ref mut ret) => Some(ret),
            _ => None,
        }
    }
}
