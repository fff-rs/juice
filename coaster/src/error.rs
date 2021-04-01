//! Defines the general set of error types in Coaster.

#[derive(Debug, thiserror::Error)]
/// Defines the set of available Coaster error types.
pub enum Error {
    /// Failure related to the Framework implementation.
    #[error("Framework error")]
    Framework(#[from] crate::framework::Error),
    /// Failure related to the Tensor.
    #[error("Tensor error")]
    Tensor(#[from] crate::tensor::Error),
    /// Failure at Plugin Operation.
    #[error("Tensor error")]
    Plugin(#[from] crate::plugin::Error),
    /// Failure related to a Device.
    #[error("Device error")]
    Device(#[from] crate::device::Error),
}
