//! Provides the INn Plugin trait for Collenchyma implementation.

use super::binary::INnBinary;
use super::operation::*;
use co::plugin::numeric_helpers::Float;
use co::binary::IBinary;
use co::tensor::SharedTensor;
use co::device::DeviceType;
use co::plugin::Error as LibError;

/// Provides the functionality for a backend to support Neural Network related operations.
pub trait INn<F: Float> {
    /// The Binary representation for this Plugin.
    type B: INnBinary<F> + IBinary;

    /// Computes the [Sigmoid function][sigmoid] over the input Tensor `x` with complete memory management.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `sigmoid_plain`.
    fn sigmoid(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the Sigmoid function over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `sigmoid`.
    fn sigmoid_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the first derivative of a [Sigmoid function][sigmoid] over the input Tensor `x` with complete memory management.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `sigmoid_diff_plain`.
    fn sigmoid_diff(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the first derivative of a Sigmoid function over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `sigmoid_diff`.
    fn sigmoid_diff_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;


    /// Returns the binary representation
    fn binary(&self) -> &Self::B;

    /// Returns the device representation
    fn device(&self) -> &DeviceType;
}
