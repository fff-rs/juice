//! Provides Rust Errors for CUDA's cuDNN status.

use std::{error, fmt};

#[derive(Debug, Copy, Clone)]
/// Defines CUDA's cuDNN errors.
pub enum Error {
    /// Failure with CUDA cuDNN initialization.
    NotInitialized(&'static str),
    /// Failure with allocation.
    AllocFailed(&'static str),
    /// Failure with a provided parameter.
    BadParam(&'static str),
    /// Failure with cuDNN.
    InternalError(&'static str),
    /// Failure with provided value.
    InvalidValue(&'static str),
    /// Failure with the hardware architecture.
    ArchMismatch(&'static str),
    /// Failure with memory access or internal error/bug.
    MappingError(&'static str),
    /// Failure with Kernel execution.
    ExecutionFailed(&'static str),
    /// Failure with an unsupported request.
    NotSupported(&'static str),
    /// Failure CUDA License.
    LicenseError(&'static str),
    /// Failure
    Unknown(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::NotInitialized(ref err) => write!(f, "{:?}", err),
            Error::AllocFailed(ref err) => write!(f, "{:?}", err),
            Error::BadParam(ref err) => write!(f, "{:?}", err),
            Error::InternalError(ref err) => write!(f, "{:?}", err),
            Error::InvalidValue(ref err) => write!(f, "{:?}", err),
            Error::ArchMismatch(ref err) => write!(f, "{:?}", err),
            Error::MappingError(ref err) => write!(f, "{:?}", err),
            Error::ExecutionFailed(ref err) => write!(f, "{:?}", err),
            Error::NotSupported(ref err) => write!(f, "{:?}", err),
            Error::LicenseError(ref err) => write!(f, "{:?}", err),
            Error::Unknown(ref err) => write!(f, "{:?}", err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::NotInitialized(ref err) => err,
            Error::AllocFailed(ref err) => err,
            Error::BadParam(ref err) => err,
            Error::InternalError(ref err) => err,
            Error::InvalidValue(ref err) => err,
            Error::ArchMismatch(ref err) => err,
            Error::MappingError(ref err) => err,
            Error::ExecutionFailed(ref err) => err,
            Error::NotSupported(ref err) => err,
            Error::LicenseError(ref err) => err,
            Error::Unknown(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&dyn error::Error> {
        match *self {
            Error::NotInitialized(_) => None,
            Error::AllocFailed(_) => None,
            Error::BadParam(_) => None,
            Error::InternalError(_) => None,
            Error::InvalidValue(_) => None,
            Error::ArchMismatch(_) => None,
            Error::MappingError(_) => None,
            Error::ExecutionFailed(_) => None,
            Error::NotSupported(_) => None,
            Error::LicenseError(_) => None,
            Error::Unknown(_) => None,
        }
    }
}
