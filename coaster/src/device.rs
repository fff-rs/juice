//! Provides a representation for one or many ready to use hardwares.
//!
//! Devices are a set of hardwares, which were initialized by the framework, in the order that they
//! were ready to receive kernel executions, event processing, memory synchronization, etc. You can
//! turn available hardware into a device, through the [backend][backend].
//!
//! [backend]: ../backend/index.html
use std::any::Any;

use crate::hardware::IHardware;
#[cfg(feature = "native")]
use crate::frameworks::native::Error as NativeError;
#[cfg(feature = "opencl")]
use frameworks::opencl::Error as OpenCLError;
#[cfg(feature = "cuda")]
use crate::frameworks::cuda::DriverError as CudaError;
use std::{fmt, error};

/// Marker trait for backing memory.
pub trait IMemory { }

/// Specifies Hardware behavior across frameworks.
pub trait IDevice
    where Self: Any + Clone + Eq + Any + MemorySync {

    /// The Hardware representation for this Device.
    type H: IHardware;
    /// The Memory representation for this Device.
    type M: IMemory + Any;
    /// Returns the unique identifier of the Device.
    fn id(&self) -> &isize;
    /// Returns the hardwares, which define the Device.
    fn hardwares(&self) -> &Vec<Self::H>;
    /// Allocate memory on the Device.
    fn alloc_memory(&self, size: usize) -> Result<Self::M, Error>;
}

/// This trait should be implemented for `Device`.
/// Use of `Any` everywhere is ugly, but it looks like there is no other way
/// to do it if we want to extract CUDA stuff into its own crate completely,
/// so that base crate knows nothing about it at all.
pub trait MemorySync {
    /// FIXME
    fn sync_in(&self, my_memory: &mut dyn Any, src_device: &dyn Any, src_memory: &dyn Any)
               -> Result<(), Error>;
    /// FIXME
    fn sync_out(&self, my_memory: &dyn Any, dst_device: &dyn Any, dst_memory: &mut dyn Any)
                -> Result<(), Error>;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Defines a generic set of Memory Errors.
pub enum Error {
    /// No route found for memory transfer between devices
    NoMemorySyncRoute,
    /// Framework error at memory synchronization.
    MemorySyncError,
    /// Framework error at memory allocation.
    MemoryAllocationError,

    /// Failures related to the Native framework implementation.
    #[cfg(feature = "native")]
    Native(NativeError),
    /// Failures related to the OpenCL framework implementation.
    #[cfg(feature = "opencl")]
    OpenCL(OpenCLError),
    /// Failures related to the Cuda framework implementation.
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::NoMemorySyncRoute => write!(f, "{}", self.to_string()),
            Error::MemorySyncError => write!(f, "{}", self.to_string()),
            Error::MemoryAllocationError => write!(f, "{}", self.to_string()),

            #[cfg(feature = "native")]
            Error::Native(ref err) => write!(f, "Native error: {}", err),
            #[cfg(feature = "opencl")]
            Error::OpenCL(ref err) => write!(f, "OpenCL error: {}", err),
            #[cfg(feature = "cuda")]
            Error::Cuda(ref err) => write!(f, "Cuda error: {}", err),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Error::NoMemorySyncRoute => None,
            Error::MemorySyncError => None,
            Error::MemoryAllocationError => None,

            #[cfg(feature = "native")]
            Error::Native(ref err) => Some(err),
            #[cfg(feature = "opencl")]
            Error::OpenCL(ref err) => Some(err),
            #[cfg(feature = "cuda")]
            Error::Cuda(ref err) => Some(err),
        }
    }
}

#[cfg(feature = "native")]
impl From<NativeError> for Error {
    fn from(err: NativeError) -> Error {
        Error::Native(err)
    }
}

#[cfg(feature = "opencl")]
impl From<OpenCLError> for Error {
    fn from(err: OpenCLError) -> Error {
        Error::OpenCL(err)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaError> for Error {
    fn from(err: CudaError) -> Error {
        Error::Cuda(err)
    }
}
