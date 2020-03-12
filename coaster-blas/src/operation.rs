//! Provides the IOperationX operation traits for Coaster's Framework implementation.

use crate::transpose::Transpose;
use coaster::plugin::Error;
use coaster::tensor::SharedTensor;

/// Describes a Asum Operation.
pub trait IOperationAsum<F> {
    /// Computes the Asum operation.
    fn compute(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), Error>;
}

/// Describes a Axpy Operation.
pub trait IOperationAxpy<F> {
    /// Computes the Axpy operation.
    fn compute(
        &self,
        a: &SharedTensor<F>,
        x: &SharedTensor<F>,
        y: &mut SharedTensor<F>,
    ) -> Result<(), Error>;
}

/// Describes a Copy Operation.
pub trait IOperationCopy<F> {
    /// Computes the Copy operation.
    fn compute(&self, x: &SharedTensor<F>, y: &mut SharedTensor<F>) -> Result<(), Error>;
}

/// Describes a Dot Operation.
pub trait IOperationDot<F> {
    /// Computes the Dot operation.
    fn compute(
        &self,
        x: &SharedTensor<F>,
        y: &SharedTensor<F>,
        result: &mut SharedTensor<F>,
    ) -> Result<(), Error>;
}

/// Describes a Nrm2 Operation.
pub trait IOperationNrm2<F> {
    /// Computes the Nrm2 operation.
    fn compute(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), Error>;
}

/// Describes a Scale Operation.
pub trait IOperationScale<F> {
    /// Computes the Scale operation.
    fn compute(&self, a: &SharedTensor<F>, x: &mut SharedTensor<F>) -> Result<(), Error>;
}

/// Describes a Swap Operation.
pub trait IOperationSwap<F> {
    /// Computes the Swap operation.
    fn compute(&self, x: &mut SharedTensor<F>, y: &mut SharedTensor<F>) -> Result<(), Error>;
}

/// Describes a Gemm Operation.
pub trait IOperationGemm<F> {
    /// Computes the Gemm operation.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::many_single_char_names)]
    fn compute(
        &self,
        alpha: &SharedTensor<F>,
        at: Transpose,
        a_dims: &[usize],
        a: &SharedTensor<F>,
        bt: Transpose,
        b_dims: &[usize],
        b: &SharedTensor<F>,
        beta: &SharedTensor<F>,
        c_dims: &[usize],
        c: &mut SharedTensor<F>,
    ) -> Result<(), ::coaster::error::Error>;
}
