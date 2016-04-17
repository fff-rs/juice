//! Provides a representation for one or many ready to use hardwares.
//!
//! Devices are a set of hardwares, which were initialized by the framework, in the order that they
//! were ready to receive kernel executions, event processing, memory synchronization, etc. You can
//! turn available hardware into a device, through the [backend][backend].
//!
//! [backend]: ../backend/index.html

use hardware::IHardware;
use memory::{IMemory, MemoryType};
#[cfg(feature = "native")]
use frameworks::native::device::Cpu;
#[cfg(feature = "native")]
use frameworks::native::Error as NativeError;
#[cfg(feature = "opencl")]
use frameworks::opencl::Context as OpenCLContext;
#[cfg(feature = "opencl")]
use frameworks::opencl::Error as OpenCLError;
#[cfg(feature = "cuda")]
use frameworks::cuda::Context as CudaContext;
#[cfg(feature = "cuda")]
use frameworks::cuda::DriverError as CudaError;
use std::{fmt, error};

/// Specifies Hardware behavior across frameworks.
pub trait IDevice {
    /// The Hardware representation for this Device.
    type H: IHardware;
    /// The Memory representation for this Device.
    type M: IMemory;
    /// Returns the unique identifier of the Device.
    fn id(&self) -> &isize;
    /// Returns the hardwares, which define the Device.
    fn hardwares(&self) -> &Vec<Self::H>;
    /// Allocate memory on the Device.
    fn alloc_memory(&self, size: usize) -> Result<Self::M, Error>;
    /// Synchronize memory from `source_data` to the memory at `dest_data`.
    ///
    /// Defines how data is synchronized into the device.
    /// All Frameworks, except Native(host), are also defining a `sync_out` method.
    fn sync_in(&self, source: &DeviceType, source_data: &MemoryType, dest_data: &mut Self::M) -> Result<(), Error>;
}

/// Specifies Sync out behavior across frameworks.
pub trait IDeviceSyncOut<T: IMemory> {
    /// The Memory representation for this Device.
    type M: IMemory;
    /// Synchronizes memory from `source_data` to `dest_data`.
    fn sync_out(&self, source_data: &Self::M, dest_data: &mut T) -> Result<(), Error>;
}


#[derive(Debug, Eq, PartialEq, Hash, Clone)]
/// Container for all known IDevice implementations
pub enum DeviceType {
    /// A native CPU
    #[cfg(feature = "native")]
    Native(Cpu),
    /// A OpenCL Context
    #[cfg(feature = "opencl")]
    OpenCL(OpenCLContext),
    /// A Cuda Context
    #[cfg(feature = "cuda")]
    Cuda(CudaContext),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
/// Defines a generic set of Memory Errors.
pub enum Error {
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
    fn description(&self) -> &str {
        match *self {
            #[cfg(feature = "native")]
            Error::Native(ref err) => err.description(),
            #[cfg(feature = "opencl")]
            Error::OpenCL(ref err) => err.description(),
            #[cfg(feature = "cuda")]
            Error::Cuda(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
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

impl From<Error> for ::tensor::Error {
    fn from(err: Error) -> ::tensor::Error {
        ::tensor::Error::MemoryAllocationError(err)
    }
}
