//! Provides a representation for one or many ready to use hardwares.
//!
//! Devices are a set of hardwares, which got initialized from the framework, in order that they
//! are ready to receive kernel executions, event processing, memory synchronization, etc. You can
//! turn available hardware into a device, through the [backend][backend].
//!
//! [backend]: ../backend/index.html

use hardware::IHardware;
use memory::{IMemory, MemoryType};
use frameworks::native::device::Cpu;
use frameworks::opencl::Context as OpenCLContext;
use frameworks::cuda::Context as CudaContext;
use frameworks::native::Error as NativeError;
use frameworks::opencl::Error as OpenCLError;
use frameworks::cuda::Error as CudaError;
use std::{fmt, error};

/// Specifies Hardware behavior accross frameworks.
pub trait IDevice {
    /// The Hardware representation for this Device.
    type H: IHardware;
    /// The Memory representation for this Device.
    type M: IMemory;
    /// Returns the unique identifier of the Device.
    fn id(&self) -> &isize;
    /// Returns the hardwares, which define the Device.
    fn hardwares(&self) -> Vec<Self::H>;
    /// Allocate memory on the Device.
    fn alloc_memory(&self, size: usize) -> Result<Self::M, Error>;
    /// Synchronize memory from this Device to `dest_device`.
    fn sync_memory_to(&self, source: &Self::M, dest: &mut MemoryType, dest_device: &DeviceType);
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
/// Container for all known IDevice implementations
pub enum DeviceType {
    /// A native CPU
    Native(Cpu),
    /// A OpenCL Context
    OpenCL(OpenCLContext),
    /// A Cuda Context
    Cuda(CudaContext),
}

#[derive(Debug, Copy, Clone)]
/// Defines a generic set of Memory Errors.
pub enum Error {
    /// Failures related to the Native framework implementation.
    Native(NativeError),
    /// Failures related to the OpenCL framework implementation.
    OpenCL(OpenCLError),
    /// Failures related to the Cuda framework implementation.
    Cuda(CudaError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Native(ref err) => write!(f, "Native error: {}", err),
            Error::OpenCL(ref err) => write!(f, "OpenCL error: {}", err),
            Error::Cuda(ref err) => write!(f, "Cuda error: {}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Native(ref err) => err.description(),
            Error::OpenCL(ref err) => err.description(),
            Error::Cuda(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::Native(ref err) => Some(err),
            Error::OpenCL(ref err) => Some(err),
            Error::Cuda(ref err) => Some(err),
        }
    }
}

impl From<NativeError> for Error {
    fn from(err: NativeError) -> Error {
        Error::Native(err)
    }
}

impl From<OpenCLError> for Error {
    fn from(err: OpenCLError) -> Error {
        Error::OpenCL(err)
    }
}

impl From<CudaError> for Error {
    fn from(err: CudaError) -> Error {
        Error::Cuda(err)
    }
}

impl From<Error> for ::shared_memory::Error {
    fn from(err: Error) -> ::shared_memory::Error {
        ::shared_memory::Error::MemoryAllocationError(err)
    }
}
