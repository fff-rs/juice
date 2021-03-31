//! Provides the interface for running parallel computations on one ore many devices.
//!
//! This is the abstraction over which you are interacting with your devices. You can create a
//! backend for computation by first choosing a specific [Framework][frameworks] such as OpenCL and
//! afterwards selecting one or many available hardwares to create a backend.
//!
//! A backend provides you with the functionality of managing the memory of the devices and copying
//! your objects from host to devices and the other way around. Additionally you can execute
//! operations in parallel through kernel functions on the device(s) of the backend.
//!
//! ## Architecture
//!
//! The initialization of a backend happens through the BackendConfig, which defines which
//! [framework][framework] should be used and which [programs][program] should be available for
//! parallel execution.
//!
//! [frameworks]: ../frameworks/index.html
//! [framework]: ../framework/index.html
//! [program]: ../program/index.html
//!
//! ## Examples
//!
//! ```
//! extern crate coaster as co;
//! use co::framework::*;
//! use co::backend::{Backend, BackendConfig};
//! use co::frameworks::Native;
//! #[allow(unused_variables)]
//! fn main() {
//!     // Initialize a new Framewok.
//!     let framework = Native::new();
//!     // After initialization, the available hardware through the Framework can be obtained.
//!     let hardwares = &framework.hardwares().to_vec();
//!     // Create a Backend configuration with
//!     // - a Framework and
//!     // - the available hardwares you would like to use for computation (turn into a device).
//!     let backend_config = BackendConfig::new(framework, hardwares);
//!     // Create a ready to go backend from the configuration.
//!     let backend = Backend::new(backend_config);
//! }
//! ```

use crate::device::IDevice;
use crate::error::Error;
use crate::framework::IFramework;

#[derive(Debug, Clone)]
/// Defines the main and highest struct of Coaster.
pub struct Backend<F: IFramework> {
    /// Provides the Framework.
    ///
    /// The Framework implementation such as OpenCL, CUDA, etc. defines, which should be used and
    /// determines which hardwares will be available and how parallel kernel functions can be
    /// executed.
    ///
    /// Default: [Native][native]
    ///
    /// [native]: ../frameworks/native/index.html
    pub framework: Box<F>,
    /// Provides a device, created from one or many hardwares, which are ready to execute kernel
    /// methods and synchronize memory.
    device: F::D,
}

/// Defines the functionality of the Backend.
impl<F: IFramework + Clone> Backend<F> {
    /// Initialize a new native Backend from a BackendConfig.
    pub fn new(config: BackendConfig<F>) -> Result<Backend<F>, Error> {
        let device = config.framework.new_device(config.hardwares)?;
        Ok(Backend {
            framework: Box::new(config.framework),
            device,
        })
    }

    /// Returns the available hardware.
    pub fn hardwares(&self) -> &[F::H] {
        self.framework.hardwares()
    }

    /// Returns the backend framework.
    pub fn framework(&self) -> &F {
        &self.framework
    }

    /// Returns the backend device.
    pub fn device(&self) -> &F::D {
        &self.device
    }
}

/// Describes a Backend.
///
/// Serves as a marker trait and helps for extern implementation.
pub trait IBackend
where
    <<Self as IBackend>::F as IFramework>::D: IDevice,
{
    /// Represents the Framework of a Backend.
    type F: IFramework + Clone;

    /// Returns the backend device.
    fn device(&self) -> &<<Self as IBackend>::F as IFramework>::D;

    /// Try to create a default backend.
    fn default() -> Result<Backend<Self::F>, Error>
    where
        Self: Sized,
    {
        let hw_framework = Self::F::new();
        let hardwares = hw_framework.hardwares();
        let framework = Self::F::new(); // dirty dirty hack to get around borrowing
        let backend_config = BackendConfig::new(framework, hardwares);
        Backend::new(backend_config)
    }

    /// Synchronize backend.
    fn synchronize(&self) -> Result<(), crate::framework::Error> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Provides Backend Configuration.
///
/// Use it to initialize a new Backend.
pub struct BackendConfig<'a, F: IFramework + 'a> {
    /// Framework - i.e. CUDA
    framework: F,
    hardwares: &'a [F::H],
}

impl<'a, F: IFramework + Clone> BackendConfig<'a, F> {
    /// Creates a new BackendConfig.
    pub fn new(framework: F, hardwares: &'a [F::H]) -> BackendConfig<'a, F> {
        BackendConfig {
            framework,
            hardwares,
        }
    }
}
