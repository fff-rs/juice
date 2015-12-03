/// Defines a generic set of Native Errors.

use std::{fmt, error};

#[derive(Debug, Copy, Clone)]
/// Defines the Native Error.
pub enum Error {
    /// Failure related to allocation, syncing memory
    Memory(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::Memory(ref err) => write!(f, "{}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Memory(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match *self {
            Error::Memory(_) => None,
        }
    }
}
