//! Provides the interface for running parallel computations on one ore many devices.
//!
//! This is the abstraction over which you are interacting with your devices. You can create a
//! backend for computation by first choosing a specifc [Framework][frameworks] such as OpenCL and
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
//! extern crate collenchyma as co;
//! use co::framework::*;
//! use co::backend::{Backend, BackendConfig};
//! use co::frameworks::Native;
//! #[allow(unused_variables)]
//! fn main() {
//!     // Initialize a new Framewok.
//!     let framework = Native::new();
//!     // After initialization, the available hardware through the Framework can be obtained.
//!     let hardwares = framework.hardwares();
//!     // Create a Backend configuration with
//!     // - a Framework and
//!     // - the available hardwares you would like to use for computation (turn into a device).
//!     let backend_config = BackendConfig::new(framework, hardwares);
//!     // Create a ready to go backend from the configuration.
//!     let backend = Backend::new(backend_config);
//! }
//! ```

use error::Error;
use framework::IFramework;
use device::{IDevice, DeviceType};

#[derive(Debug, Clone)]
/// Defines the main and highest struct of Collenchyma.
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
    framework: Box<F>,
    /// Provides a device, created from one or many hardwares, which are ready to execute kernel
    /// methods and synchronize memory.
    device: DeviceType,
}

/// Defines the functionality of the Backend.
impl<F: IFramework + Clone> Backend<F> {
    /// Initialize a new native Backend from a BackendConfig.
    pub fn new(config: BackendConfig<F>) -> Result<Backend<F>, Error> {
        let device = try!(config.framework.new_device(config.hardwares));
        Ok(
            Backend {
                framework: Box::new(config.framework),
                device: device,
            }
        )
    }

    /// Returns the available hardware.
    pub fn hardwares(&self) -> Vec<F::H> {
        self.framework.hardwares()
    }

    /// Returns the backend framework.
    pub fn framework(&self) -> &Box<F> {
        &self.framework
    }

    /// Returns the backend device.
    pub fn device(&self) -> &DeviceType {
        &self.device
    }
}

/// Describes a Backend.
///
/// Serves as a marker trait and helps for extern implementation.
pub trait IBackend {
    /// Represents the Framework of a Backend.
    type F: IFramework + Clone;

    /// Returns the backend device.
    fn device(&self) -> &DeviceType;
}

#[derive(Debug, Clone)]
/// Provides Backend Configuration.
///
/// Use it to initialize a new Backend.
pub struct BackendConfig<F: IFramework> {
    framework: F,
    hardwares: Vec<F::H>,
}

impl<F: IFramework + Clone> BackendConfig<F> {
    /// Creates a new BackendConfig.
    pub fn new(framework: F, hardwares: Vec<F::H>) -> BackendConfig<F> {
        BackendConfig {
            framework: framework.clone(),
            hardwares: hardwares,
        }
    }
}
