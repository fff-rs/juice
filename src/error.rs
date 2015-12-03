//! Defines the general set of error types in Collenchyma.

use std::{error, fmt};

#[derive(Debug)]
/// Defines the set of available Collenchyma error types.
pub enum Error {
    /// Failure related to the Framework implementation.
    Framework(::framework::Error),
    /// Failure related to the SharedMemory.
    SharedMemory(::shared_memory::Error),
    /// Failure related to an Library(Operation).
    Operation(::libraries::Error),
    /// Failure related to a Device.
    Device(::device::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Framework(ref err) => write!(f, "Framwork error: {}", err),
            Error::SharedMemory(ref err) => write!(f, "SharedMemory error: {}", err),
            Error::Operation(ref err) => write!(f, "Library/Operation error: {}", err),
            Error::Device(ref err) => write!(f, "Device error: {}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Framework(ref err) => err.description(),
            Error::SharedMemory(ref err) => err.description(),
            Error::Operation(ref err) => err.description(),
            Error::Device(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::Framework(ref err) => Some(err),
            Error::SharedMemory(ref err) => Some(err),
            Error::Operation(ref err) => Some(err),
            Error::Device(ref err) => Some(err),
        }
    }
}
