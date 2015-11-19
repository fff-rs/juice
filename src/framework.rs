//! Provides the generic functionality of a device supporting frameworks such as Host CPU, OpenCL, CUDA,
//! etc..
//! [device]: ../device/index.html
//!
//! The default Framework would be plain host CPU for common computation. To make use of other
//! computation devices such as GPUs you would choose other computation Frameworks such as OpenCL
//! or CUDA, which provide the access to those devices for computation.
//!
//! To start backend-agnostic and highly parallel computation, you start by initializing on of the
//! Framework implementations, resulting in an initialized Framework, that contains among
//! other things, a list of all available devices through that framework.
//!
//! ## Examples
//!
//! ```
//! // Initializing a Framework
//! // let framework: Framework = OpenCL::new();
//! // let backend: Backend = framework.create_backend();
//! ```

use device::Device;

#[derive(Debug, Clone)]
/// The Framework
pub struct Framework {
    devices: Vec<Device>,
}

/// Defines a Framework.
pub trait IFramework {

    /// Defines the Framework by a Name.
    ///
    /// For convention, let the ID be uppercase.<br/>
    /// EXAMPLE: OPENCL
    const ID: &'static str;

    /// Initializes a new Framework.
    ///
    /// Loads all the available devices
    fn new() -> Self where Self: Sized;

    /// Provices all the available devices.
    fn load_devices() -> Vec<Device>;
}

#[derive(Debug)]
/// Defines a generic struct for Framework Error.
///
/// Is used as a Err for Framework Results.
pub struct FrameworkError {
    /// The returned value from the ff.
    pub code: i32,
    /// The returned error ID from the ff.
    pub id: String,
    /// The message, that should be returned.
    pub message: String,
}
