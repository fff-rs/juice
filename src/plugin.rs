//! Provides the IBlas library trait for Collenchyma implementation.

use super::binary::IBlasBinary;
use super::transpose::*;
use collenchyma::binary::IBinary;
use collenchyma::tensor::SharedTensor;
use collenchyma::device::DeviceType;

/// Provides the functionality for a backend to support Basic Linear Algebra Subprogram operations.
pub trait IBlas<F> { }

/// Provides the asum operation.
pub trait Asum<F> {
    /// Computes the absolute sum of vector `x`.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    fn asum(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
            -> Result<(), ::collenchyma::error::Error>;
}

/// Provides the axpy operation.
pub trait Axpy<F> {
    /// Computes a vector `x` times a constant `a` plus a vector `y` aka. `a * x + y`.
    ///
    /// Saves the resulting vector back into `y`.
    /// This is a Level 1 BLAS operation.
    fn axpy(&self, a: &SharedTensor<F>, x: &SharedTensor<F>, y: &mut SharedTensor<F>)
            -> Result<(), ::collenchyma::error::Error>;
}

/// Provides the copy operation.
pub trait Copy<F> {
    /// Copies `x.len()` elements of vector `x` into vector `y`.
    ///
    /// Saves the result to `y`.
    /// This is a Level 1 BLAS operation.
    fn copy(&self, x: &SharedTensor<F>, y: &mut SharedTensor<F>)
            -> Result<(), ::collenchyma::error::Error>;
}

/// Provides the dot operation.
pub trait Dot<F> {
    /// Computes the [dot product][dot-product] over x and y.
    /// [dot-product]: https://en.wikipedia.org/wiki/Dot_product
    ///
    /// Saves the resulting value into `result`.
    /// This is a Level 1 BLAS operation.
    fn dot(&self, x: &SharedTensor<F>, y: &SharedTensor<F>,
           result: &mut SharedTensor<F>)
           -> Result<(), ::collenchyma::error::Error>;
}

/// Provides the nrm2 operation.
pub trait Nrm2<F> {
    /// Computes the L2 norm aka. euclidean length of vector `x`.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    fn nrm2(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
            -> Result<(), ::collenchyma::error::Error>;
}

/// Provides the scal operation.
pub trait Scal<F> {
    /// Scales a vector `x` by a constant `a` aka. `a * x`.
    ///
    /// Saves the resulting vector back into `x`.
    /// This is a Level 1 BLAS operation.
    fn scal(&self, a: &SharedTensor<F>, x: &mut SharedTensor<F>)
            -> Result<(), ::collenchyma::error::Error>;
}

/// Provides the swap operation.
pub trait Swap<F> {
    /// Swaps the content of vector `x` and vector `y` with complete memory management.
    ///
    /// Saves the resulting vector back into `x`.
    /// This is a Level 1 BLAS operation.
    fn swap(&self, x: &mut SharedTensor<F>, y: &mut SharedTensor<F>)
            -> Result<(), ::collenchyma::error::Error>;
}

/// Provides the gemm operation.
pub trait Gemm<F> {
    /// Computes a matrix-matrix product with general matrices.
    ///
    /// Saves the result into `c`.
    /// This is a Level 3 BLAS operation.
    fn gemm(&self, alpha: &SharedTensor<F>,
            at: Transpose, a: &SharedTensor<F>,
            bt: Transpose, b: &SharedTensor<F>,
            beta: &SharedTensor<F>,
            c: &mut SharedTensor<F>) -> Result<(), ::collenchyma::error::Error>;
}

/// Allows a BlasBinary to be provided which is used for a IBlas implementation.
pub trait BlasBinaryProvider<F, B: IBlasBinary<F> + IBinary> {
    /// Returns the binary representation
    fn binary(&self) -> &B;
    /// Returns the device representation
    fn device(&self) -> &DeviceType;
}

impl<F, B: IBlasBinary<F> + IBinary> IBlas<F> for BlasBinaryProvider<F, B> { }
