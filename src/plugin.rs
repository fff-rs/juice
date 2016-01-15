//! Provides the INn Plugin trait for Collenchyma implementation.

use co::plugin::numeric_helpers::Float;
use co::tensor::SharedTensor;
use co::device::DeviceType;

/// Provides generic NN Operation Config functionality.
///
/// Needs to be implemented for Operation specific configurations.
pub trait NNOperationConfig<F> {}

/// Provides the functionality for a backend to support Neural Network related operations.
pub trait NN<F: Float> {
    /// The Convolution Operation Config representation for this Plugin.
    type CC: NNOperationConfig<F>;
    /// The LRN Operation Config representation for this Plugin.
    type CLRN: NNOperationConfig<F>;
    /// The Pooling Operation Config representation for this Plugin.
    type CPOOL: NNOperationConfig<F>;

    /// Initializes the Plugin.
    fn init_nn();

    /// Returns the device on which the Plugin operations will run.
    fn device(&self) -> &DeviceType;
}

/// Provides the functionality for a Backend to support Sigmoid operations.
pub trait Sigmoid<F: Float> : NN<F> {
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
}

/// Provides the functionality for a Backend to support ReLU operations.
pub trait Relu<F: Float> : NN<F> {
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
}

/// Provides the functionality for a Backend to support TanH operations.
pub trait Tanh<F: Float> : NN<F> {
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
}

/// Provides the functionality for a Backend to support Convolution operations.
pub trait Convolution<F: Float> : NN<F> {
    /// Creates a new ConvolutionConfig, which needs to be passed to further convolution Operations.
    fn new_convolution_config(&self, src: &SharedTensor<F>, dest: &SharedTensor<F>, filter: &mut SharedTensor<F>, stride: &[i32], zero_padding: &[i32]) -> Result<Self::CC, ::co::error::Error>;

    /// Computes a [CNN convolution][convolution] over the input Tensor `x` with complete memory management.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `convolution_plain`.
    fn convolution(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the convolution over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `convolution`.
    fn convolution_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [CNN convolution][convolution] over the input Tensor `x` with complete memory management.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `convolution_grad_plain`.
    fn convolution_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a convolution over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `convolution_grad`.
    fn convolution_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>, config: &Self::CC) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Softmax operations.
pub trait Softmax<F: Float> : NN<F> {
    /// Computes a [Softmax activation][softmax] over the input Tensor `x` with complete memory management.
    /// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `softmax_plain`.
    fn softmax(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the softmax over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `softmax`.
    fn softmax_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [Softmax activation][softmax] over the input Tensor `x` with complete memory management.
    /// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `softmax_grad_plain`.
    fn softmax_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a softmax over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `softmax_grad`.
    fn softmax_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Local Response Normalization operations.
pub trait LRN<F: Float> : NN<F> {
    /// Creates a new (Local Response Normalization) LRNConfig, which needs to be passed to further LRN Operations.
    fn new_lrn_config(&self, n: u32, alpha: f64, beta: f64, k: f64) -> Result<Self::CLRN, ::co::error::Error>;

    /// Computes a [LRN][lrn] over the input Tensor `x` with complete memory management.
    /// [lrn]: https://en.wikipedia.org/wiki/lrnal_neural_network
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `lrn_plain`.
    fn lrn(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>, config: &Self::CLRN) -> Result<(), ::co::error::Error>;

    /// Computes the LRN over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `lrn`.
    fn lrn_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>, config: &Self::CLRN) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [LRN][lrn] over the input Tensor `x` with complete memory management.
    /// [lrn]: https://en.wikipedia.org/wiki/lrnal_neural_network
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `lrn_grad_plain`.
    fn lrn_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>, config: &Self::CLRN) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a LRN over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `lrn_grad`.
    fn lrn_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>, config: &Self::CLRN) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Pooling operations.
pub trait Pooling<F: Float> : NN<F> {
    /// Creates a new PoolingConfig, which needs to be passed to further pooling Operations.
    fn new_pooling_config(&self, window: &[i32], padding: &[i32], stride: &[i32]) -> Result<Self::CPOOL, ::co::error::Error>;

    /// Computes non-linear down-sampling ([max Pooling][pooling]) over the input Tensor `x` with complete memory management.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `pooling_max_plain`.
    fn pooling_max(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>, config: &Self::CPOOL) -> Result<(), ::co::error::Error>;

    /// Computes the max pooling over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `pooling_max`.
    fn pooling_max_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>, config: &Self::CPOOL) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [max Pooling][pooling] over the input Tensor `x` with complete memory management.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `pooling_max_grad_plain`.
    fn pooling_max_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>, config: &Self::CPOOL) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of max pooling over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `pooling_max_grad`.
    fn pooling_max_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>, config: &Self::CPOOL) -> Result<(), ::co::error::Error>;
}
