//! Provides the IBlasBinary binary trait for Collenchyma's Framework implementation.

use super::operation::*;
use collenchyma::plugin::numeric_helpers::Float;

/// Describes the operation binding for a Blas Binary implementation.
pub trait IBlasBinary<F: Float> {
    /// Describes the Asum Operation.
    type Asum: IOperationAsum<F >;
    /// Describes the Axpy Operation.
    type Axpy: IOperationAxpy<F>;
    /// Describes the Copy Operation.
    type Copy: IOperationCopy<F>;
    /// Describes the Dot Operation.
    type Dot: IOperationDot<F>;
    /// Describes the Nrm2 Operation.
    type Nrm2: IOperationNrm2<F>;
    /// Describes the Scale Operation.
    type Scale: IOperationScale<F>;
    /// Describes the Swap Operation.
    type Swap: IOperationSwap<F>;

    /// Returns an initialized Asum operation.
    fn asum(&self) -> Self::Asum;
    /// Returns an initialized Axpy operation.
    fn axpy(&self) -> Self::Axpy;
    /// Returns an initialized Copy operation.
    fn copy(&self) -> Self::Copy;
    /// Returns an initialized Dot operation.
    fn dot(&self) -> Self::Dot;
    /// Returns an initialized Nrm2 operation.
    fn nrm2(&self) -> Self::Nrm2;
    /// Returns an initialized Scale operation.
    fn scale(&self) -> Self::Scale;
    /// Returns an initialized Swap operation.
    fn swap(&self) -> Self::Swap;
}
