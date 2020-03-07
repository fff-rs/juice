//! Provides the generic functionality of a hardware supporting frameworks such as native CPU, OpenCL,
//! CUDA, etc..
//! [hardware]: ../hardware/index.html
//!
//! The default Framework would be a plain host CPU for common computation. To make use of other
//! computation hardwares such as GPUs you would choose other computation Frameworks such as OpenCL
//! or CUDA, which provide the access to those hardwares for computation.
//!
//! To start backend-agnostic and highly parallel computation, you start by initializing one of the
//! Framework implementations, resulting in an initialized Framework, that contains, among
//! other things, a list of all available hardwares through that framework.
//!
//! ## Examples
//!
//! ```
//! // Initializing a Framework
//! // let framework: Framework = OpenCL::new();
//! // let backend: Backend = framework.create_backend();
//! ```

use crate::hardware::IHardware;
use crate::device::IDevice;
use crate::binary::IBinary;
#[cfg(feature = "opencl")]
use frameworks::opencl::Error as OpenCLError;
#[cfg(feature = "cuda")]
use crate::frameworks::cuda::DriverError as CudaError;
use std::error;
use std::fmt;

/// Defines a Framework.
pub trait IFramework {
    /// The Hardware representation for this Framework.
    type H: IHardware;
    /// The Device representation for this Framework.
    type D: IDevice + Clone;
    /// The Binary representation for this Framework.
    type B: IBinary + Clone;

    /// Defines the Framework by a Name.
    ///
    /// For convention, let the ID be uppercase.<br/>
    /// EXAMPLE: OPENCL
    #[allow(non_snake_case)]
    fn ID() -> &'static str;

    /// Initializes a new Framework.
    ///
    /// Loads all the available hardwares
    fn new() -> Self where Self: Sized;

    /// Initializes all the available hardwares.
    fn load_hardwares() -> Result<Vec<Self::H>, Error>;

    /// Returns the cached and available hardwares.
    fn hardwares(&self) -> &[Self::H];

    /// Returns the initialized binary.
    fn binary(&self) -> &Self::B;

    /// Initializes a new Device from the provided hardwares.
    fn new_device(&self, _: &[Self::H]) -> Result<Self::D, Error>;
}

#[derive(Debug)]
/// Defines a generic set of Framework Errors.
pub enum Error {
    /// Failures related to the OpenCL framework implementation.
    #[cfg(feature = "opencl")]
    OpenCL(OpenCLError),
    /// Failures related to the Cuda framework implementation.
    #[cfg(feature = "cuda")]
    Cuda(CudaError),
    /// Failure related to the Coaster implementation of a specific Framework.
    Implementation(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            #[cfg(feature = "opencl")]
            Error::OpenCL(ref err) => write!(f, "OpenCL error: {}", err),
            #[cfg(feature = "cuda")]
            Error::Cuda(ref err) => write!(f, "Cuda error: {}", err),
            Error::Implementation(ref err) => write!(f, "Coaster Implementation error: {}", err),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            #[cfg(feature = "opencl")]
            Error::OpenCL(ref err) => Some(err),
            #[cfg(feature = "cuda")]
            Error::Cuda(ref err) => Some(err),
            Error::Implementation(_) => None,
        }
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

impl From<Error> for crate::error::Error {
    fn from(err: Error) -> crate::error::Error {
        crate::error::Error::Framework(err)
    }
}
