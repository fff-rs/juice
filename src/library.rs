//! Provides the IBlas library trait for Collenchyma implementation.

use super::binary::IBlasBinary;
use super::operation::*;
use collenchyma::plugin::numeric_helpers::Float;
use collenchyma::binary::IBinary;
use collenchyma::shared_memory::{SharedMemory, TensorR0, TensorR1};
use collenchyma::device::DeviceType;
use collenchyma::plugin::Error as LibError;

/// Provides the functionality for a backend to support Basic Linear Algebra Subprogram operations.
pub trait IBlas<F: Float> {
    /// The Binary representation for this Library.
    type B: IBlasBinary<F> + IBinary;

    /// Computes the absolute sum of vector `x` with complete memory management.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `asum_plain`.
    fn asum(&self, x: &mut SharedMemory<F, TensorR1>, result: &mut SharedMemory<F, TensorR0>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().asum().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(result.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Computes the absolute sum of vector `x` without any memory management.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `asum`.
    fn asum_plain(&self, x: &mut SharedMemory<F, TensorR1>, result: &mut SharedMemory<F, TensorR0>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().asum().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(result.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Computes a vector `x` times a constant `a` plus a vector `y` aka. `a * x + y` with complete memory management.
    ///
    /// Saves the resulting vector back into `y`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `axpy_plain`.
    fn axpy(&self, a: &mut SharedMemory<F, TensorR0>, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
        Ok(try!(
            self.binary().axpy().compute(
                try!(a.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
            )
        ))
    }

    /// Computes a vector `x` times a constant `a` plus a vector `y` aka. `a * x + y` without any memory management.
    ///
    /// Saves the resulting vector back into `y`.
    /// This is a Level 1 BLAS operation.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `axpy`.
    fn axpy_plain(&self, a: &mut SharedMemory<F, TensorR0>, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().axpy().compute(
                try!(a.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
            )
        ))
    }

    /// Copies `x.len()` elements of vector `x` into vector `y` with complete memory management.
    ///
    /// Saves the result to `y`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `copy_plain`.
    fn copy(&self, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match y.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().copy().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
            )
        ))
    }

    /// Copies `x.len()` elements of vector `x` into vector `y` without any memory management.
    ///
    /// Saves the result to `y`.
    /// This is a Level 1 BLAS operation.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `copy`.
    fn copy_plain(&self, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().copy().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
            )
        ))
    }

    /// Computes the [dot product][dot-product] over x and y with complete memory management.
    /// [dot-product]: https://en.wikipedia.org/wiki/Dot_product
    ///
    /// Saves the resulting value into `result`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `dot_plain`.
    fn dot(&self, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>, result: &mut SharedMemory<F, TensorR0>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().dot().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                try!(result.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Computes the [dot product][dot-product] over x and y without any memory management.
    /// [dot-product]: https://en.wikipedia.org/wiki/Dot_product
    ///
    /// Saves the resulting value into `result`.
    /// This is a Level 1 BLAS operation.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `dot`.
    fn dot_plain(&self, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>, result: &mut SharedMemory<F, TensorR0>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().dot().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
                try!(result.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Computes the L2 norm aka. euclidean length of vector `x` with complete memory management.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `nrm2_plain`.
    fn nrm2(&self, x: &mut SharedMemory<F, TensorR1>, result: &mut SharedMemory<F, TensorR0>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().nrm2().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(result.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Computes the L2 norm aka. euclidean length of vector `x` without any memory management.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `nrm2`.
    fn nrm2_plain(&self, x: &mut SharedMemory<F, TensorR1>, result: &mut SharedMemory<F, TensorR0>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().nrm2().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(result.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Scales a vector `x` by a constant `a` aka. `a * x` with complete memory management.
    ///
    /// Saves the resulting vector back into `x`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `scale_plain`.
    fn scale(&self, a: &mut SharedMemory<F, TensorR0>, x: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        Ok(try!(
            self.binary().scale().compute(
                try!(a.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                try!(x.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
            )
        ))
    }

    /// Scales a vector `x` by a constant `a` aka. `a * x` without any memory management.
    ///
    /// Saves the resulting vector back into `x`.
    /// This is a Level 1 BLAS operation.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `scale`.
    fn scale_plain(&self, a: &mut SharedMemory<F, TensorR0>, x: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().scale().compute(
                try!(a.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `a`"))),
                try!(x.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
            )
        ))
    }

    /// Swaps the content of vector `x` and vector `y` with complete memory management.
    ///
    /// Saves the resulting vector back into `x`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `swap_plain`.
    fn swap(&self, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
        Ok(try!(
            self.binary().swap().compute(
                try!(x.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
            )
        ))
    }

    /// Swaps the content of vector `x` and vector `y` without any memory management.
    ///
    /// Saves the resulting vector back into `x`.
    /// This is a Level 1 BLAS operation.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `swap`.
    fn swap_plain(&self, x: &mut SharedMemory<F, TensorR1>, y: &mut SharedMemory<F, TensorR1>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().swap().compute(
                try!(x.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(y.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `y`"))),
            )
        ))
    }

    /// Returns the binary representation
    fn binary(&self) -> &Self::B;

    /// Returns the device representation
    fn device(&self) -> &DeviceType;
}
