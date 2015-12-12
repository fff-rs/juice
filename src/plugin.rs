//! Provides the INn Plugin trait for Collenchyma implementation.

use super::binary::INnBinary;
use super::operation::*;
use collenchyma::plugin::numeric_helpers::Float;
use collenchyma::binary::IBinary;
use collenchyma::tensor::SharedTensor;
use collenchyma::device::DeviceType;
use collenchyma::plugin::Error as LibError;

/// Provides the functionality for a backend to support Neural Network related operations.
pub trait INn<F: Float> {
    /// The Binary representation for this Plugin.
    type B: INnBinary<F> + IBinary;

    /// Computes the absolute sum of vector `x` with complete memory management.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    ///
    /// For a no-memory managed version see `asum_plain`.
    fn sigmoid(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::collenchyma::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().sigmoid().compute(
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
    fn sigmoid_plain(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::collenchyma::error::Error> {
        Ok(try!(
            self.binary().sigmoid().compute(
                try!(x.get(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `x`"))),
                try!(result.get_mut(self.device()).ok_or(LibError::MissingMemoryForDevice("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Returns the binary representation
    fn binary(&self) -> &Self::B;

    /// Returns the device representation
    fn device(&self) -> &DeviceType;
}
