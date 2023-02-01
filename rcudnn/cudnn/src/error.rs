//! Provides Rust Errors for CUDA's cuDNN status.

#[allow(unused)]
pub type Result<T> = std::result::Result<T, Error>;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, thiserror::Error)]
/// Defines CUDA's cuDNN errors.
pub enum Error {
    /// Failure with CUDA cuDNN initialization.
    #[error("{0:?}")]
    NotInitialized(&'static str),
    /// Failure with allocation.
    #[error("{0:?}")]
    AllocFailed(&'static str),
    /// Failure with a provided parameter.
    #[error("{0:?}")]
    BadParam(&'static str),
    /// Failure with cuDNN.
    #[error("{0:?}")]
    InternalError(&'static str),
    /// Failure with provided value.
    #[error("{0:?}")]
    InvalidValue(&'static str),
    /// Failure with the hardware architecture.
    #[error("{0:?}")]
    ArchMismatch(&'static str),
    /// Failure with memory access or internal error/bug.
    #[error("{0:?}")]
    MappingError(&'static str),
    /// Failure with Kernel execution.
    #[error("{0:?}")]
    ExecutionFailed(&'static str),
    /// Failure with an unsupported request.
    #[error("{0:?}")]
    NotSupported(&'static str),
    /// Failure CUDA License.
    #[error("{0:?}")]
    LicenseError(&'static str),
    /// Failure
    #[error("{0:?}: {1}")]
    Unknown(&'static str, u64),
}
