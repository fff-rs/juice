//! Provides the IOperationX operation traits for Collenchyma's Framework implementation.

use collenchyma::plugin::numeric_helpers::Float;
use collenchyma::memory::MemoryType;
use collenchyma::plugin::Error;

/// Describes a Sigmoid Operation.
pub trait IOperationSigmoid<F: Float> {
    /// Computes the Sigmoid operation.
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}
