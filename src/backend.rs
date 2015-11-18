//! Provides the interface for running parallel computations on one ore many devices.
//!
//! This is the abstraction over which you are interacting with your devices. You can create a
//! backend for computation by first choosing a specifc [Framework][frameworks] such as OpenCL and
//! afterwards selecting one or many devices to create a backend.
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
//! use co::framework::{IFramework};
//! use co::backend::*;
//! use co::frameworks::host::Host;
//! fn main() {
//!     let backend = Backend::<Host, SingleDevice>::new(Host, Host::load_devices(), SingleDevice);
//! }
//! ```

use framework::{IFramework, FrameworkError};
use frameworks::Host;
use device::Device;

#[derive(Debug, Copy, Clone)]
/// Defines a single device backend.
pub struct SingleDevice;
#[derive(Debug, Copy, Clone)]
/// Defines a multi device backend.
pub struct MultiDevice;

/// Defines the parent backend type.
pub trait IBackendType { }
/// Defines the single backend type.
pub trait ISingleDevice { }
/// Defines the multi backend type.
pub trait IMultiDevice { }

impl IBackendType for SingleDevice { }
impl ISingleDevice for SingleDevice { }

impl IBackendType for MultiDevice { }
impl IMultiDevice for MultiDevice { }

#[derive(Debug, Clone)]
/// Defines the main and highest struct of Collenchyma.
pub struct Backend<F: IFramework, T: IBackendType> {
    /// Provides the Framework.
    ///
    /// The Framework implementation such as OpenCL, CUDA, etc. defines, which should be used and
    /// determines which devices will be available and how parallel kernel functions can be
    /// executed.
    ///
    /// Default: [Host][host]
    ///
    /// [host]: ../frameworks/host/index.html
    framework: Box<F>,
    /// Provides all the devices that are available through the Framework.
    devices: Vec<Device>,
    backend_type: T,
}

/// Defines the functionality of the Backend.
impl<F: IFramework + Clone, T: IBackendType> Backend<F, T> {

    /// Initialize a new Backend with a specific Framework
    ///
    /// Loads all the available devices through the Framework
    pub fn new(framework: F, devices: Vec<Device>, b_type: T) -> Backend<F, T> {
        if devices.len() > 1 {
            Backend {
                framework: Box::new(framework.clone()),
                devices: devices,
                backend_type: b_type,
            }
        } else {
            Backend {
                framework: Box::new(framework.clone()),
                devices: devices,
                backend_type: b_type,
            }
        }
    }
}

impl<F: IFramework, T: ISingleDevice> Backend<F, T> {

    /// Executes a kernel on the backend.
    ///
    /// This is the main function of the Collenchyma. It takes care of syncing the memory to the
    /// backend device, where the operation will be executed, executes the operation in parallel if
    /// so desired and returns the result.
    pub fn call() -> i32 {
        unimplemented!()
    }
}

impl<F, T> Backend<F, T>
    where F: IFramework,
          T: IMultiDevice {

    /// Executes a kernel on the backend.
    ///
    /// This is the main function of the Collenchyma. It takes care of syncing the memory to the
    /// backend device, where the operation will be executed, executes the operation in parallel if
    /// so desired and returns the result.
    pub fn call() -> i32 {
        unimplemented!()
    }
}
