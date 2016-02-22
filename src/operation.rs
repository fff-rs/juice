//! Provides the IOperationX operation traits for Collenchyma's Framework implementation.

use collenchyma::memory::MemoryType;
use collenchyma::plugin::Error;
use ::transpose::Transpose;

/// Describes a Asum Operation.
pub trait IOperationAsum<F> {
    /// Computes the Asum operation.
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Axpy Operation.
pub trait IOperationAxpy<F> {
    /// Computes the Axpy operation.
    fn compute(&self, a: &MemoryType, x: & MemoryType, y: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Copy Operation.
pub trait IOperationCopy<F> {
    /// Computes the Copy operation.
    fn compute(&self, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Dot Operation.
pub trait IOperationDot<F> {
    /// Computes the Dot operation.
    fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Nrm2 Operation.
pub trait IOperationNrm2<F> {
    /// Computes the Nrm2 operation.
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Scale Operation.
pub trait IOperationScale<F> {
    /// Computes the Scale operation.
    fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Swap Operation.
pub trait IOperationSwap<F> {
    /// Computes the Swap operation.
    fn compute(&self, x: &mut MemoryType, y: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Gemm Operation.
pub trait IOperationGemm<F> {
    /// Computes the Gemm operation.
    fn compute(&self, alpha: &MemoryType, at: Transpose, a_dims: &[usize], a: &MemoryType, bt: Transpose, b_dims: &[usize], b: &MemoryType, beta: &MemoryType, c_dims: &[usize], c: &mut MemoryType) -> Result<(), ::collenchyma::error::Error>;
}
