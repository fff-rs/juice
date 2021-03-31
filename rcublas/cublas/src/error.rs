//! Provides Rust Errors for every cuBLAS status.

#[allow(unused)]
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Copy, Clone, thiserror::Error)]
/// Defines cuBLAS errors.
pub enum Error {
    /// Failure with cuBLAS initialization.
    #[error("CUDA Driver/Runtime API not initialized.")]
    NotInitialized,
    /// Failure with allocation.
    #[error("The resources could not be allocated.")]
    AllocFailed,
    /// Failure with cuDNN.
    #[error("Internal: {0}")]
    InternalError(&'static str),
    /// Failure with provided value.
    #[error("Invalid value: {0}")]
    InvalidValue(&'static str),
    /// Failure with the hardware architecture.
    #[error("cuBLAS only supports devices with compute capabilities greater than or equal to 1.3.")]
    ArchMismatch,
    /// Failure with memory access or internal error/bug.
    #[error("There was an error accessing GPU memory.")]
    MappingError,
    /// Failure with Kernel execution.
    #[error("Execution failed to launch on the GPU.")]
    ExecutionFailed,
    /// Failure with an unsupported request.
    #[error("Not supported: {0}")]
    NotSupported(&'static str),
    /// Failure CUDA License.
    #[error("There is an error with the license. Check that it is present, unexpired and the NVIDIA_LICENSE_FILE environment variable has been set correctly.")]
    LicenseError,
    /// Failure
    #[error("Unknown error: {0} - code {1}")]
    Unknown(&'static str, u64),
}
