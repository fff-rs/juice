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
//! [frameworks]: ../frameworks/index.html

use framework::FrameworkError;

#[derive(Debug, Copy, Clone)]
/// Defines the main and highest struct of Collenchyma.
pub struct Buffer;

/// Defines the functionality of a Backend.
pub trait IBuffer {

}
