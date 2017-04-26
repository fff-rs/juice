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
    /// Compute the convolution as Fast-Fourier Transform with 32x32 tiles.
    ///
    /// Needs a significant memory workspace.
    FFTTiling,
    /// Compute the convolution without implicit or explicit matrix-multiplication. **Do not try to use this**.
    ///
    /// Listed in cuDNN docs but cuDNN does not provide a implementation.
    Direct,
    /// Winograd  Transform
    Winograd,
    /// Winograd  Transform Non-Fused
    WinogradNonFused,
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
    /// Winograd  Transform Non-Fused
    WinogradNonFused,
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
    /// Compute the convolution as Fast-Fourier Transform with 32x32 tiles.
    ///
    /// Needs a significant memory workspace.
    ///
    /// The results are deterministic.
    FFTTiling,
    /// Winograd  Transform
    Winograd,
    /// Winograd  Transform Non-Fused
    WinogradNonFused,
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
    /// The Activation Operation Config representation for this Plugin.

    /// Initializes the Plugin.
    fn init_nn();

    /// Returns the device on which the Plugin operations will run.
    fn device(&self) -> &DeviceType;
}

/// Provides the functionality for a Backend to support Sigmoid operations.
pub trait Sigmoid<F> : NN<F> {
    /// Computes the [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result to `result`.
    fn sigmoid(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
               -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result to `result_diff`.
    fn sigmoid_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                    result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>)
                    -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for pointwise Sigmoid operations (overwrites the input with the result of the operation).
pub trait SigmoidPointwise<F> : NN<F> {
    /// Computes the [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result back to `x`.
    ///
    /// For a no-memory managed version see `sigmoid_pointwise_plain`.
    fn sigmoid_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result back to `x_diff`.
    fn sigmoid_pointwise_grad(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support ReLU operations.
pub trait Relu<F> : NN<F> {
    /// Computes the [Rectified linear units][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result to `result`.
    fn relu(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [ReLU][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result to `result_diff`.
    fn relu_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                 result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>)
                 -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for pointwise ReLU operations (overwrites the input with the result of the operation).
pub trait ReluPointwise<F> : NN<F> {
    /// Computes the [Rectified linear units][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result back to `x`.
    fn relu_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [ReLU][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result back to `x_diff`.
    fn relu_pointwise_grad(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>)
                           -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support TanH operations.
pub trait Tanh<F> : NN<F> {
    /// Computes the [hyperbolic Tangent][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result to `result`.
    fn tanh(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
            -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [tanh][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result to `result_diff`.
    fn tanh_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                 result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>)
                 -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for pointwise ReLU operations (overwrites the input
/// with the result of the operation).
pub trait TanhPointwise<F> : NN<F> {
    /// Computes the [hyperbolic Tangent][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result back to `x`.
    fn tanh_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [tanh][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result back to `x_diff`.
    fn tanh_pointwise_grad(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>)
                           -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Convolution operations.
pub trait Convolution<F> : NN<F> {
    /// Creates a new ConvolutionConfig, which needs to be passed to further
    /// convolution Operations.
    fn new_convolution_config(&self,
                              src: &SharedTensor<F>,
                              dest: &SharedTensor<F>,
                              filter: &SharedTensor<F>,
                              algo_fwd: ConvForwardAlgo,
                              algo_bwd_filter: ConvBackwardFilterAlgo,
                              algo_bwd_data: ConvBackwardDataAlgo,
                              stride: &[i32],
                              zero_padding: &[i32])
                              -> Result<Self::CC, ::co::error::Error>;

    /// Computes a [CNN convolution][convolution] over the input Tensor `x`.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `result`.
    fn convolution(&self,
                   filter: &SharedTensor<F>,
                   x: &SharedTensor<F>,
                   result: &mut SharedTensor<F>,
                   workspace: &mut SharedTensor<u8>,
                   config: &Self::CC)
                   -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [CNN convolution][convolution] with respect to the filter.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `filter_diff`.
    fn convolution_grad_filter(&self,
                               src_data: &SharedTensor<F>,
                               dest_diff: &SharedTensor<F>,
                               filter_diff: &mut SharedTensor<F>,
                               workspace: &mut SharedTensor<u8>,
                               config: &Self::CC)
                               -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [CNN convolution][convolution] over the input
    /// Tensor `x` with respect to the data.
    /// [convolution]: https://en.wikipedia.org/wiki/Convolutional_neural_network
    ///
    /// Saves the result to `result_diff`.
    fn convolution_grad_data(&self,
                             filter: &SharedTensor<F>,
                             x_diff: &SharedTensor<F>,
                             result_diff: &mut SharedTensor<F>,
                             workspace: &mut SharedTensor<u8>,
                             config: &Self::CC)
                             -> Result<(), ::co::error::Error>;

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
}

/// Provides the functionality for a Backend to support Softmax operations.
pub trait Softmax<F> : NN<F> {
    /// Computes a [Softmax][softmax] over the input Tensor `x`.
    /// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
    ///
    /// Saves the result to `result`.
    fn softmax(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
               -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [Softmax][softmax] over the input Tensor `x`.
    /// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
    ///
    /// Saves the result to `result_diff`.
    fn softmax_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                    result_diff: &mut SharedTensor<F>)
                    -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support LogSoftmax operations.
pub trait LogSoftmax<F> : NN<F> {
    /// Computes a logarithmic softmax over the input Tensor `x`.
    ///
    /// Saves the result to `result`.
    fn log_softmax(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
                   -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a logarithmic softmax over the input Tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    fn log_softmax_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                        result_diff: &mut SharedTensor<F>)
                        -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Local Response Normalization operations.
pub trait LRN<F> : NN<F> {
    /// Creates a new (Local Response Normalization) LRNConfig, which needs to be
    /// passed to further LRN Operations.
    fn new_lrn_config(&self, n: u32, alpha: f64, beta: f64, k: f64)
                      -> Result<Self::CLRN, ::co::error::Error>;

    /// Computes a [LRN][lrn] over the input Tensor `x`.
    /// [lrn]: https://en.wikipedia.org/wiki/lrnal_neural_network
    ///
    /// Saves the result to `result`.
    fn lrn(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>,
           config: &Self::CLRN) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of a [LRN][lrn] over the input Tensor `x`.
    /// [lrn]: https://en.wikipedia.org/wiki/lrnal_neural_network
    ///
    /// Saves the result to `result_diff`.
    fn lrn_grad(&self,
                x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>,
                config: &Self::CLRN)
                -> Result<(), ::co::error::Error>;
}

/// Provides the functionality for a Backend to support Pooling operations.
pub trait Pooling<F> : NN<F> {
    /// Creates a new PoolingConfig, which needs to be passed to further pooling Operations.
    fn new_pooling_config(&self, window: &[i32], stride: &[i32], padding: &[i32])
                          -> Result<Self::CPOOL, ::co::error::Error>;

    /// Computes non-linear down-sampling ([max Pooling][pooling]) over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result`.
    fn pooling_max(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>,
                   config: &Self::CPOOL) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [max Pooling][pooling] over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result_diff`.
    fn pooling_max_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                        result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>,
                        config: &Self::CPOOL) -> Result<(), ::co::error::Error>;


    /// Computes non-linear down-sampling ([average Pooling][pooling]) over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result`.
    fn pooling_avg(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>,
                   config: &Self::CPOOL) -> Result<(), ::co::error::Error>;

    /// Computes the gradient of [average Pooling][pooling] over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result_diff`.
    fn pooling_avg_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                        result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>,
                        config: &Self::CPOOL) -> Result<(), ::co::error::Error>;
}
