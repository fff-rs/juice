//! Provides the INn Plugin trait for Collenchyma implementation.

use co::tensor::SharedTensor;
use co::device::DeviceType;

#[derive(Debug, Copy, Clone)]
/// Different algorithms to compute the convolution forward algorithm.
pub enum ConvForwardAlgo {
    /// Attempt to automatically find the best algorithm of all the other available ones.
    Auto,
    /// Compute the convolution as explicit matrix product.
    ///
    /// Needs a significant memory workspace.
    GEMM,
    /// Compute the convolution as matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ImplicitGEMM,
    /// Similar to `ImplicitGEMM` but needs some workspace to precompile the implicit indices.
    ImplicitPrecompiledGEMM,
    /// Compute the convolution as Fast-Fourier Transform.
    ///
    /// Needs a significant memory workspace.
    FFT,
    /// Compute the convolution without implicit or explicit matrix-multiplication. **Do not try to use this**.
    ///
    /// Listed in cuDNN docs but cuDNN does not provide a implementation.
    Direct,
}

impl ConvForwardAlgo {
    /// Check if algorithim should be chosen automatically.
    pub fn is_auto(&self) -> bool {
        match *self {
            ConvForwardAlgo::Auto => true,
            _ => false
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Different algorithms to compute the gradient with respect to the filter.
pub enum ConvBackwardFilterAlgo {
    /// Attempt to automatically find the best algorithm of all the other available ones.
    Auto,
    /// Compute the convolution as matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are deterministic.
    ImplicitGEMM,
    /// Compute the convolution as sum of matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are non-deterministic.
    ImplicitGEMMSum,
    /// Similar to `ImplicitGEMMSum` but needs some workspace to precompile the implicit indices.
    ///
    /// The results are non-deterministic.
    ImplicitPrecompiledGEMMSum,
    /// Compute the convolution as Fast-Fourier Transform.
    ///
    /// Needs a significant memory workspace.
    ///
    /// The results are deterministic.
    FFT,
}

impl ConvBackwardFilterAlgo {
    /// Check if algorithim should be chosen automatically.
    pub fn is_auto(&self) -> bool {
        match *self {
            ConvBackwardFilterAlgo::Auto => true,
            _ => false
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Different algorithms to compute the gradient with respect to the filter.
pub enum ConvBackwardDataAlgo {
    /// Attempt to automatically find the best algorithm of all the other available ones.
    Auto,
    /// Compute the convolution as matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are deterministic.
    ImplicitGEMM,
    /// Compute the convolution as sum of matrix product without forming the matrix that holds the input data.
    ///
    /// Does not need any memory workspace.
    ///
    /// The results are non-deterministic.
    ImplicitGEMMSum,
    /// Compute the convolution as Fast-Fourier Transform.
    ///
    /// Needs a significant memory workspace.
    ///
    /// The results are deterministic.
    FFT,
}

impl ConvBackwardDataAlgo {
    /// Check if algorithim should be chosen automatically.
    pub fn is_auto(&self) -> bool {
        match *self {
            ConvBackwardDataAlgo::Auto => true,
            _ => false
        }
    }
}

/// Provides generic NN Operation Config functionality.
///
/// Needs to be implemented for Operation specific configurations.
pub trait NNOperationConfig<F> {}

/// Provides Convolution Config functionality.
///
/// Needs to be implemented for Operation specific configurations.
pub trait ConvolutionConfig<F> {
    /// Returns the largest workspace size in bytes needed
    /// for any of the convolution operations.
    fn workspace_size(&self) -> usize {
        0
    }
}

/// Provides the functionality for a backend to support Neural Network related operations.
pub trait NN<F> {
    /// The Convolution Operation Config representation for this Plugin.
    type CC: NNOperationConfig<F> + ConvolutionConfig<F>;
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
pub trait Sigmoid<F> : NN<F> {
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

/// Provides the functionality for pointwise Sigmoid operations (overwrites the input with the result of the operation).
pub trait SigmoidPointwise<F> : NN<F> {
    /// Computes the [Sigmoid function][sigmoid] over the input Tensor `x` with complete memory management.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result back to `x`.
    ///
    /// For a no-memory managed version see `sigmoid_pointwise_plain`.
    fn sigmoid_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the Sigmoid function over the input Tensor `x` without any memory management.
    ///
    /// Saves the result back to `x`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `sigmoid_pointwise`.
    fn sigmoid_pointwise_plain(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [Sigmoid function][sigmoid] over the input Tensor `x` with complete memory management.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// For a no-memory managed version see `sigmoid_pointwise_grad_plain`.
    fn sigmoid_pointwise_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a Sigmoid function over the input Tensor `x` without any memory management.
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `sigmoid_pointwise_grad`.
    fn sigmoid_pointwise_grad_plain(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support ReLU operations.
pub trait Relu<F> : NN<F> {
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

/// Provides the functionality for pointwise ReLU operations (overwrites the input with the result of the operation).
pub trait ReluPointwise<F> : NN<F> {
    /// Computes the [Rectified linear units][relu] over the input Tensor `x` with complete memory management.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result back to `x`.
    ///
    /// For a no-memory managed version see `relu_pointwise_plain`.
    fn relu_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the ReLU over the input Tensor `x` without any memory management.
    ///
    /// Saves the result back to `x`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `relu_pointwise`.
    fn relu_pointwise_plain(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [ReLU][relu] over the input Tensor `x` with complete memory management.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// For a no-memory managed version see `relu_pointwise_grad_plain`.
    fn relu_pointwise_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of ReLU over the input Tensor `x` without any memory management.
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `relu_pointwise_grad`.
    fn relu_pointwise_grad_plain(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support TanH operations.
pub trait Tanh<F> : NN<F> {
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

/// Provides the functionality for pointwise ReLU operations (overwrites the input with the result of the operation).
pub trait TanhPointwise<F> : NN<F> {
    /// Computes the [hyperbolic Tangent][tanh] over the input Tensor `x` with complete memory management.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result back to `x`.
    ///
    /// For a no-memory managed version see `tanh_pointwise_plain`.
    fn tanh_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the tanh over the input Tensor `x` without any memory management.
    ///
    /// Saves the result back to `x`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `tanh_pointwise`.
    fn tanh_pointwise_plain(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [tanh][tanh] over the input Tensor `x` with complete memory management.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// For a no-memory managed version see `tanh_pointwise_grad_plain`.
    fn tanh_pointwise_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of tanh over the input Tensor `x` without any memory management.
    ///
    /// Saves the result back to `x_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `tanh_pointwise_grad`.
    fn tanh_pointwise_grad_plain(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Convolution operations.
pub trait Convolution<F> : NN<F> {
    /// Creates a new ConvolutionConfig, which needs to be passed to further convolution Operations.
    fn new_convolution_config(&self, src: &SharedTensor<F>, dest: &SharedTensor<F>, filter: &mut SharedTensor<F>,
                            algo_fwd: ConvForwardAlgo, algo_bwd_filter: ConvBackwardFilterAlgo, algo_bwd_data: ConvBackwardDataAlgo,
                            stride: &[i32], zero_padding: &[i32]) -> Result<Self::CC, ::co::error::Error>;

    /// Computes a [CNN convolution][convolution] over the input Tensor `x` with complete memory management.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `convolution_plain`.
    fn convolution(&self, filter: &mut SharedTensor<F>, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>, workspace: &mut SharedTensor<u8>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the convolution over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `convolution`.
    fn convolution_plain(&self, filter: &SharedTensor<F>, x: &SharedTensor<F>, result: &mut SharedTensor<F>, workspace: &mut SharedTensor<u8>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [CNN convolution][convolution] with respect to the filter and complete memory management.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `filter_diff`.
    ///
    /// For a no-memory managed version see `convolution_grad_filter_plain`.
    fn convolution_grad_filter(&self, src_data: &mut SharedTensor<F>, dest_diff: &mut SharedTensor<F>, filter_diff: &mut SharedTensor<F>, workspace: &mut SharedTensor<u8>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a convolution with respect to the filter and without any memory management.
    ///
    /// Saves the result to `filter_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `convolution_grad_filter`.
    fn convolution_grad_filter_plain(&self, src_data: &SharedTensor<F>, dest_diff: &SharedTensor<F>, filter_diff: &mut SharedTensor<F>, workspace: &mut SharedTensor<u8>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [CNN convolution][convolution] over the input Tensor `x` with respect to the data and complete memory management.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `convolution_grad_data_plain`.
    fn convolution_grad_data(&self, filter: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>, workspace: &mut SharedTensor<u8>, config: &Self::CC) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a convolution over the input Tensor `x` with respect to the data and without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `convolution_grad_data`.
    fn convolution_grad_data_plain(&self, filter: &SharedTensor<F>, x_diff: &SharedTensor<F>, result_diff: &mut SharedTensor<F>, workspace: &mut SharedTensor<u8>, config: &Self::CC) -> Result<(), ::co::error::Error>;
}

    // /// Computes the backward Convolution function w.r.t the bias.
    // ///
    // /// Writes the result of the computation to `bias_data`.
    // pub fn convolution_backward_bias<T>(
    //     &self,
    //     dest_grad_desc: &TensorDescriptor,
    //     dest_grad_data: *const ::libc::c_void,
    //     bias_desc: &TensorDescriptor,
    //     bias_data: *mut ::libc::c_void,
    //     scale: ScalParams<T>,
    // }
    //
    // /// Computes the backward Convolution function w.r.t the filter.
    // ///
    // /// Writes the result of the computation to `filter_data`.
    // pub fn convolution_backward_filter<T>(
    //     &self,
    //     conv_config: &ConvolutionConfig,
    //     src_desc: &TensorDescriptor,
    //     src_data: *const ::libc::c_void,
    //     dest_grad_desc: &TensorDescriptor,
    //     dest_grad_data: *const ::libc::c_void,
    //     filter_data: *mut ::libc::c_void,
    //     scale: ScalParams<T>,
    // }

/// Provides the functionality for a Backend to support Softmax operations.
pub trait Softmax<F> : NN<F> {
    /// Computes a [Softmax][softmax] over the input Tensor `x` with complete memory management.
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

    /// Computes the gradient of a [Softmax][softmax] over the input Tensor `x` with complete memory management.
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

/// Provides the functionality for a Backend to support LogSoftmax operations.
pub trait LogSoftmax<F> : NN<F> {
    /// Computes a logarithmic softmax over the input Tensor `x` with complete memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// For a no-memory managed version see `log_softmax_plain`.
    fn log_softmax(&self, x: &mut SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the logarithmic softmax over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `log_softmax`.
    fn log_softmax_plain(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a logarithmic softmax over the input Tensor `x` with complete memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// For a no-memory managed version see `log_softmax_grad_plain`.
    fn log_softmax_grad(&self, x: &mut SharedTensor<F>, x_diff: &mut SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a logarithmic softmax over the input Tensor `x` without any memory management.
    ///
    /// Saves the result to `result_diff`.
    ///
    /// *Attention*:<br/>
    /// For a correct computation result, you need to manage the memory allocation and synchronization yourself.<br/>
    /// For a memory managed version see `log_softmax_grad`.
    fn log_softmax_grad_plain(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>, result_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Local Response Normalization operations.
pub trait LRN<F> : NN<F> {
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
pub trait Pooling<F> : NN<F> {
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
