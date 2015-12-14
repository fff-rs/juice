//! Provides the INn Plugin trait for Collenchyma implementation.

use super::binary::INnBinary;
use co::plugin::numeric_helpers::Float;
use co::binary::IBinary;
use co::tensor::SharedTensor;
use co::device::DeviceType;

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

    /// Computes the gradient of a [Sigmoid function][sigmoid] over the input Tensor `x` with complete memory management.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `sigmoid_grad_plain`.
    fn sigmoid_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a Sigmoid function over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `sigmoid_grad`.
    fn sigmoid_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the [Rectified linear units][relu] over the input Tensor `x` with complete memory management.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `relu_plain`.
    fn relu(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the ReLU over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `relu`.
    fn relu_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [ReLU][relu] over the input Tensor `x` with complete memory management.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `relu_grad_plain`.
    fn relu_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of ReLU over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `relu_grad`.
    fn relu_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the [hyperbolic Tangent][tanh] over the input Tensor `x` with complete memory management.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `tanh_plain`.
    fn tanh(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the tanh over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `tanh`.
    fn tanh_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [tanh][tanh] over the input Tensor `x` with complete memory management.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `tanh_grad_plain`.
    fn tanh_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of tanh over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `tanh_grad`.
    fn tanh_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Returns the binary representation
    fn binary(&self) -> &Self::B;

    /// Returns the device representation
    fn device(&self) -> &DeviceType;
}
