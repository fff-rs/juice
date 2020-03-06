//! Provides Rust Errors for every cuBLAS status.

use std::{fmt, error};

#[derive(Debug, Copy, Clone)]
/// Defines cuBLAS errors.
pub enum Error {
    /// Failure with cuBLAS initialization.
    NotInitialized,
    /// Failure with allocation.
    AllocFailed,
    /// Failure with cuDNN.
    InternalError(&'static str),
    /// Failure with provided value.
    InvalidValue(&'static str),
    /// Failure with the hardware architecture.
    ArchMismatch,
    /// Failure with memory access or internal error/bug.
    MappingError,
    /// Failure with Kernel execution.
    ExecutionFailed,
    /// Failure with an unsupported request.
    NotSupported(&'static str),
    /// Failure CUDA License.
    LicenseError,
    /// Failure
    Unknown(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::NotInitialized => write!(f, "{:?}", "Failure with cuBLAS initialization.".to_string()),
            Error::AllocFailed => write!(f, "{:?}", "Failure with allocation.".to_string()),
            Error::InternalError(ref err) => write!(f, "{:?}", err),
            Error::InvalidValue(ref err) => write!(f, "{:?}", err),
            Error::ArchMismatch => write!(f, "{:?}", "Failure with the hardware architecture.".to_string()),
            Error::MappingError => write!(f, "{:?}", "Failure with memory access or internal error/bug.".to_string()),
            Error::ExecutionFailed => write!(f, "{:?}", "Failure with Kernel execution.".to_string()),
            Error::NotSupported(ref err) => write!(f, "{:?}", err),
            Error::LicenseError => write!(f, "{:?}", "Failure CUDA License".to_string()),
            Error::Unknown(ref err) => write!(f, "{:?}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::NotInitialized => "CUDA Driver/Runtime API not initialized.",
            Error::AllocFailed => "The resources could not be allocated.",
            Error::InternalError(ref err) => err,
            Error::InvalidValue(ref err) => err,
            Error::ArchMismatch => {
                "cuBLAS only supports devices with compute capabilities greater than or equal to 1.3."
            }
            Error::MappingError => "There was an error accessing GPU memory.",
            Error::ExecutionFailed => "Execution failed to launch on the GPU.",
            Error::NotSupported(ref err) => err,
            Error::LicenseError => {
                "There is an error with the license. Check that it is present, unexpired and the NVIDIA_LICENSE_FILE environment variable has been set correctly."
            }
            Error::Unknown(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&dyn error::Error> {
        match *self {
            Error::NotInitialized => None,
            Error::AllocFailed => None,
            Error::InternalError(_) => None,
            Error::InvalidValue(_) => None,
            Error::ArchMismatch => None,
            Error::MappingError => None,
            Error::ExecutionFailed => None,
            Error::NotSupported(_) => None,
            Error::LicenseError => None,
            Error::Unknown(_) => None,
        }
    }
}
