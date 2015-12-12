//! Provides the IOperationX operation traits for Collenchyma's Framework implementation.

use co::plugin::numeric_helpers::Float;
use co::memory::MemoryType;
use co::plugin::Error;

/// Describes a Sigmoid Operation.
pub trait IOperationSigmoid<F: Float> {
    /// Computes the Sigmoid operation.
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}
