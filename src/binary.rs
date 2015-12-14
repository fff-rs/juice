//! Provides the INnBinary Binary trait for Collenchyma's Framework implementation.

use super::operation::*;
use co::plugin::numeric_helpers::Float;

/// Describes the operation binding for a NN Binary implementation.
pub trait INnBinary<F: Float> {
    /// Describes the Sigmoid Operation.
    type Sigmoid: IOperationSigmoid<F >;

    /// Returns an initialized Sigmoid operation.
    fn sigmoid(&self) -> Self::Sigmoid;
}
