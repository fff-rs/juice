//! Provides a representation for memory across different frameworks.
//!
//! Memory is allocated by a device in a way that it is accessible for its computations.
//!
//! Normally you will want to use [SharedTensor][tensor] which handles synchronization
//! of the latest memory copy to the required device.
//!
//! [tensor]: ../tensor/index.html

#[cfg(feature = "native")]
use frameworks::native::flatbox::FlatBox;
#[cfg(feature = "opencl")]
use frameworks::opencl::memory::Memory as OpenCLMemory;
#[cfg(feature = "cuda")]
use frameworks::cuda::memory::Memory as CudaMemory;

/// Specifies Memory behavior accross frameworks.
pub trait IMemory { }

#[derive(Debug)]
/// Container for all known IMemory implementations
pub enum MemoryType {
    /// A Native Memory representation.
    #[cfg(feature = "native")]
    Native(FlatBox),
    /// A OpenCL Memory representation.
    #[cfg(feature = "opencl")]
    OpenCL(OpenCLMemory),
    /// A Cuda Memory representation.
    #[cfg(feature = "cuda")]
    Cuda(CudaMemory),
}

impl MemoryType {
    /// Extract the FlatBox if MemoryType is Native.
    #[cfg(feature = "native")]
    pub fn as_native(&self) -> Option<&FlatBox> {
        match *self {
            MemoryType::Native(ref ret) => Some(ret),
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => None,
        }
    }

    /// Extract the FlatBox mutably if MemoryType is Native.
    #[cfg(feature = "native")]
    pub fn as_mut_native(&mut self) -> Option<&mut FlatBox> {
        match *self {
            MemoryType::Native(ref mut ret) => Some(ret),
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => None,
        }
    }

    #[cfg(feature = "native")]
    /// Consumes the Memory and returns an owned OpenCL Memory.
    pub fn into_native(self) -> Option<FlatBox> {
        match self {
            MemoryType::Native(ret) => Some(ret),
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => None,
        }
    }

    /// Extract the OpenCL Memory if MemoryType is OpenCL.
    #[cfg(feature = "opencl")]
    pub fn as_opencl(&self) -> Option<&OpenCLMemory> {
        match *self {
            MemoryType::OpenCL(ref ret) => Some(ret),
            #[cfg(any(feature = "cuda", feature = "native"))]
            _ => None,
        }
    }

    /// Extract the OpenCL Memory mutably if MemoryType is OpenCL.
    #[cfg(feature = "opencl")]
    pub fn as_mut_opencl(&mut self) -> Option<&mut OpenCLMemory> {
        match *self {
            MemoryType::OpenCL(ref mut ret) => Some(ret),
            #[cfg(any(feature = "cuda", feature = "native"))]
            _ => None,
        }
    }

    #[cfg(feature = "opencl")]
    /// Consumes the Memory and returns an owned OpenCL Memory.
    pub fn into_opencl(self) -> Option<OpenCLMemory> {
        match self {
            MemoryType::OpenCL(ret) => Some(ret),
            #[cfg(any(feature = "cuda", feature = "native"))]
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    /// Extract the Cuda Memory if MemoryType is Cuda
    pub fn as_cuda(&self) -> Option<&CudaMemory> {
        match *self {
            MemoryType::Cuda(ref ret) => Some(ret),
            #[cfg(any(feature = "opencl", feature = "native"))]
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    /// Extract the Cuda Memory mutably if MemoryType is Cuda
    pub fn as_mut_cuda(&mut self) -> Option<&mut CudaMemory> {
        match *self {
            MemoryType::Cuda(ref mut ret) => Some(ret),
            #[cfg(any(feature = "opencl", feature = "native"))]
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    /// Consumes the Memory and returns an owned Cuda Memory.
    pub fn into_cuda(self) -> Option<CudaMemory> {
        match self {
            MemoryType::Cuda(ret) => Some(ret),
            #[cfg(any(feature = "opencl", feature = "native"))]
            _ => None,
        }
    }
}
