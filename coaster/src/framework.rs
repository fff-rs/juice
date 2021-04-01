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

use crate::binary::IBinary;
use crate::device::IDevice;
#[cfg(feature = "cuda")]
use crate::frameworks::cuda::DriverError as CudaError;
use crate::hardware::IHardware;
#[cfg(feature = "opencl")]
use frameworks::opencl::Error as OpenCLError;
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
    fn new() -> Self
    where
        Self: Sized;

    /// Initializes all the available hardwares.
    fn load_hardwares() -> Result<Vec<Self::H>, Error>;

    /// Returns the cached and available hardwares.
    fn hardwares(&self) -> &[Self::H];

    /// Returns the initialized binary.
    fn binary(&self) -> &Self::B;

    /// Initializes a new Device from the provided hardwares.
    fn new_device(&self, _: &[Self::H]) -> Result<Self::D, Error>;
}

#[derive(Debug, thiserror::Error)]
/// Defines a generic set of Framework Errors.
pub enum Error {
    /// Failures related to the OpenCL framework implementation.
    #[cfg_attr(feature = "opencl", error(transparent))]
    #[cfg(feature = "opencl")]
    OpenCL(#[from] OpenCLError),
    /// Failures related to the Cuda framework implementation.
    #[cfg_attr(feature = "cuda", error(transparent))]
    #[cfg(feature = "cuda")]
    Cuda(#[from] CudaError),
    /// Failure related to the Coaster implementation of a specific Framework.
    #[error("Coaster implementation error: {0}")]
    Implementation(String),
}
