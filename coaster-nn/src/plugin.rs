//! Provides the INn Plugin trait for Coaster implementation.
use crate::co::tensor::SharedTensor;
use std::fmt::Formatter;

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

/// Provides Rnn Config functionality.
///
/// Needs to be implemented for Operation specific configurations.
pub trait RnnConfig<F> {
    /// Workspace Size - Overwritten by each plugin method except native, which doesn't require
    /// a workspace size.
    fn workspace_size(&self) -> usize { 0 }
}

/// Provides the functionality for a backend to support Neural Network related operations.
pub trait NN<F> {
    /// The Convolution Operation Config representation for this Plugin.
    type CC: NNOperationConfig<F> + ConvolutionConfig<F>;
    /// The LRN Operation Config representation for this Plugin.
    type CLRN: NNOperationConfig<F>;
    /// The Pooling Operation Config representation for this Plugin.
    type CPOOL: NNOperationConfig<F>;
    // /// The Activation Operation Config representation for this Plugin.
    // type CACTI: NNOperationConfig<F>;
    /// The Dropout Operation Config representation for this Plugin.
    type CDROP: NNOperationConfig<F>;
    /// The RNN Operation Config representation for this Plugin
    type CRNN: NNOperationConfig<F> + RnnConfig<F>;

    /// Initializes the Plugin.
    fn init_nn();
}

/// Provides the functionality for a Backend to support Sigmoid operations.
pub trait Sigmoid<F> : NN<F> {
    /// Computes the [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result to `result`.
    fn sigmoid(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
               -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of a [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result to `result_diff`.
    fn sigmoid_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                    result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>)
                    -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for pointwise Sigmoid operations (overwrites the input with the result of the operation).
pub trait SigmoidPointwise<F> : NN<F> {
    /// Computes the [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result back to `x`.
    ///
    /// For a no-memory managed version see `sigmoid_pointwise_plain`.
    fn sigmoid_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of a [Sigmoid function][sigmoid] over the input Tensor `x`.
    /// [sigmoid]: https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// Saves the result back to `x_diff`.
    fn sigmoid_pointwise_grad(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>) -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for a Backend to support ReLU operations.
pub trait Relu<F> : NN<F> {
    /// Computes the [Rectified linear units][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result to `result`.
    fn relu(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>) -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of [ReLU][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result to `result_diff`.
    fn relu_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                 result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>)
                 -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for pointwise ReLU operations (overwrites the input with the result of the operation).
pub trait ReluPointwise<F> : NN<F> {
    /// Computes the [Rectified linear units][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result back to `x`.
    fn relu_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of [ReLU][relu] over the input Tensor `x`.
    /// [relu]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    ///
    /// Saves the result back to `x_diff`.
    fn relu_pointwise_grad(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>)
                           -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for a Backend to support TanH operations.
pub trait Tanh<F> : NN<F> {
    /// Computes the [hyperbolic Tangent][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result to `result`.
    fn tanh(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
            -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of [hyperbolic Tangent][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result to `result_diff`.
    fn tanh_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                 result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>)
                 -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for pointwise ReLU operations (overwrites the input
/// with the result of the operation).
pub trait TanhPointwise<F> : NN<F> {
    /// Computes the [hyperbolic Tangent][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result back to `x`.
    fn tanh_pointwise(&self, x: &mut SharedTensor<F>) -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of [tanh][tanh] over the input Tensor `x`.
    /// [tanh]: https://en.wikipedia.org/wiki/Hyperbolic_function
    ///
    /// Saves the result back to `x_diff`.
    fn tanh_pointwise_grad(&self, x: &SharedTensor<F>, x_diff: &mut SharedTensor<F>)
                           -> Result<(), crate::co::error::Error>;
}

/// Provide the functionality for a Backend to support RNN operations
pub trait Rnn<F>: NN<F> {
    /// Create a RnnConfig
    fn new_rnn_config(
        &self,
        src: &SharedTensor<F>,
        dropout_probability: Option<f32>,
        dropout_seed: Option<u64>,
        sequence_length: i32,
        network_mode: RnnNetworkMode,
        input_mode: RnnInputMode,
        direction_mode: DirectionMode,
        algorithm: RnnAlgorithm,
        hidden_size: i32,
        num_layers: i32,
        batch_size: i32,
        // RC being RNNConfig
    ) -> Result<Self::CRNN, crate::co::error::Error>;

    /// Generate Weights for RNN
    fn generate_rnn_weight_description(
        &self,
        rnn_config: &Self::CRNN,
        sequence_length: i32,
        batch_size: i32,
        input_size: i32,
    ) -> Result<Vec<Vec<usize>>, crate::co::error::Error>;

    /// Train a LSTM Network and Return Results
    // TODO: Create alternate rnn_forward or alternate path to work with pretrained networks
    /// # Arguments
    /// * `weight_desc` Previously initialised FilterDescriptor for Weights
    fn rnn_forward(
        &self,
        src: &SharedTensor<F>,
        output: &mut SharedTensor<F>,
        rnn_config: &Self::CRNN,
        weights: &[&SharedTensor<F>],
        workspace: &mut SharedTensor<u8>,
    ) -> Result<(), crate::co::error::Error>;

    /// Calculates RNN Gradients for Input/Hidden/Cell
    /// Compute Gradient of Input w.r.t. Output
    fn rnn_backward_data(&self,
                         src: &SharedTensor<F>,
                         src_gradient: &mut SharedTensor<F>,
                         output: &SharedTensor<F>,
                         output_gradient: &SharedTensor<F>,
                         rnn_config: &Self::CRNN,
                         weights: &[&SharedTensor<F>],
                         workspace: &mut SharedTensor<u8>)
                         -> Result<(), crate::co::error::Error>;

    /// Calculates RNN Gradients for Weights
    /// Compute Gradient of Weights w.r.t. Output 
    fn rnn_backward_weights(&self,
                            src: &SharedTensor<F>,
                            output: &SharedTensor<F>,
                            weight_gradients: &mut [&mut SharedTensor<F>],
                            rnn_config: &Self::CRNN,
                            workspace: &mut SharedTensor<u8>)
                            -> Result<(), crate::co::error::Error>;
}

#[derive(Debug, Copy, Clone)]
/// Network Type for RNN Networks [cudnnRNNMOde_t][1]
/// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNMode_t
pub enum RnnNetworkMode {
    /// CUDNN_RNN_RELU - Single gate RNN with a ReLU activation function
    ReLU,
    /// Single-gate RNN with a tanh activation function
    Tanh,
    /// Four-gate LSTM Network with no peephole connection
    LSTM,
    /// Three-gate network with Gated Recurrent Units
    GRU
}

impl std::fmt::Display for RnnNetworkMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result = match &self {
            RnnNetworkMode::ReLU => "RelU",
            RnnNetworkMode::Tanh => "Tanh",
            RnnNetworkMode::LSTM => "LSTM",
            RnnNetworkMode::GRU => "GRU",
        }.to_owned();
        write!(f, "{}", result)
    }
}

impl RnnNetworkMode {
    /// Convert RnnNetworkMode to String Representation
    pub fn from_string(input: &str) -> Result<Self, &str> {
        match input {
            "GRU" => Ok(RnnNetworkMode::GRU),
            "LSTM" => Ok(RnnNetworkMode::LSTM),
            "ReLU" => Ok(RnnNetworkMode::ReLU),
            "Tanh" => Ok(RnnNetworkMode::Tanh),
            _ => Err("Unknown RnnType used - variants are GRU, LSTM, ReLU, and Tanhd"),
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Input Modes for RNN [cudnnRNNInputMode_t][1]
/// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNInputMode_t
pub enum RnnInputMode {
    /// CUDNN_LINEAR_INPUT - A biased matrix multiplication is performed at the input of the first
    /// recurrent layer
    LinearInput,
    /// CUDNN_SKIP_INPUT - No operation is performed at the input of the first recurrent layer -
    /// if this is used then the leading dimension of the input tensor must be equal to the hidden
    /// state size of the network.
    SkipInput,
}

impl std::fmt::Display for RnnInputMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result = match &self {
            RnnInputMode::LinearInput => "LinearInput",
            RnnInputMode::SkipInput => "SkipInput"
        }.to_owned();
        write!(f, "{}", result)
    }
}

impl RnnInputMode {
    /// Convert to RnnInputMode from String Representation
    pub fn from_string(input: &str) -> Result<Self, &str> {
        match input {
            "LinearInput" => Ok(RnnInputMode::LinearInput),
            "SkipInput" => Ok(RnnInputMode::SkipInput),
            _ => Err("Unknown RnnInputMode used - variants are LinearInput, SkipInput"),
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Direction Mode for RNN [cudnnDirectionMode_t][1]
/// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnDirectionMode_t
pub enum DirectionMode {
    /// CUDNN_UNIDIRECTIONAL - The network iterates from first to last
    UniDirectional,
    /// CUDNN_BIDIRECTION - Concats recurrent output of First -> Last && Last -> First
    BiDirectional,
}

impl std::fmt::Display for DirectionMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result = match &self {
            DirectionMode::UniDirectional => "UniDirectional",
            DirectionMode::BiDirectional => "BiDirectional"
        }.to_owned();
        write!(f, "{}", result)
    }
}

impl DirectionMode {
    /// Convert to DirectionMode from String Representation
    pub fn from_string(input: &str) -> Result<Self, &str> {
        match input {
            "UniDirectional" => Ok(DirectionMode::UniDirectional),
            "BiDirectional" => Ok(DirectionMode::BiDirectional),
            _ => Err("Unknown DirectionMode used - variants are UniDirectional, BiDirectional"),
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Algorithm for RNN [cudnnRNNAlgo_t][1]
/// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNAlgo_t
///
/// Persist Static requires v6+
pub enum RnnAlgorithm {
    /// Sequence of Operations for each RNN Layer
    Standard,
    /// Uses a Persistent Kernel - fast when the first D of the input is small
    PersistStatic,
    /// RNN parts use a persistent kernel. Fast when the first dimension is small, and when it can
    /// reuse plans in repeated calls.
    PersistDynamic,
    /// Count - Cannot find in docs but is in Generated - FIXME
    Count,
}

impl std::fmt::Display for RnnAlgorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result = match &self {
            RnnAlgorithm::Standard => "Standard",
            RnnAlgorithm::PersistStatic => "PersistStatic",
            RnnAlgorithm::PersistDynamic => "PersistDynamic",
            RnnAlgorithm::Count => unreachable!()
        }.to_owned();
        write!(f, "{}", result)
    }
}

impl RnnAlgorithm {
    /// Convert to RnnAlgorithm from String Representation
    fn from_string(input: &str) -> Result<Self, &str> {
        match input {
            "Standard" => Ok(RnnAlgorithm::Standard),
            "PersistStatic" => Ok(RnnAlgorithm::PersistStatic),
            "PersistDynamic" => Ok(RnnAlgorithm::PersistDynamic),
            _ => Err("Unknown RnnAlgorithm used - variants are Standard, PersistStatic, PersistDynamic"),
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Enables/Disables the padded input/output [cudnnRNNPaddingMode_t][1]
/// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnRNNPaddingMode_t
pub enum RnnPaddingMode {
    /// Padding disabled
    Disabled,
    /// Padding enabled
    Enabled,
}

#[derive(Debug, Copy, Clone)]
/// Indicate if Tensor Core Operations are permitted [cudnnMathType_t][1]
/// [1]: https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnMathType_t
pub enum MathType {
    /// No Tensor Core ops
    Default,
    /// Uses Tensor Core ops
    TensorOPMath,
    /// Uses FP32 Tensors for input/output
    TensorOPMathAllowConversion
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
                              -> Result<Self::CC, crate::co::error::Error>;

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
                   -> Result<(), crate::co::error::Error>;

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
                               -> Result<(), crate::co::error::Error>;

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
                             -> Result<(), crate::co::error::Error>;

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
               -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of a [Softmax][softmax] over the input Tensor `x`.
    /// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
    ///
    /// Saves the result to `result_diff`.
    fn softmax_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                    result_diff: &mut SharedTensor<F>)
                    -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for a Backend to support LogSoftmax operations.
pub trait LogSoftmax<F> : NN<F> {
    /// Computes a logarithmic softmax over the input Tensor `x`.
    ///
    /// Saves the result to `result`.
    fn log_softmax(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>)
                   -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of a logarithmic softmax over the input Tensor `x`.
    ///
    /// Saves the result to `result_diff`.
    fn log_softmax_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                        result_diff: &mut SharedTensor<F>)
                        -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for a Backend to support Local Response Normalization operations.
pub trait LRN<F> : NN<F> {
    /// Creates a new (Local Response Normalization) LRNConfig, which needs to be
    /// passed to further LRN Operations.
    fn new_lrn_config(&self, n: u32, alpha: f64, beta: f64, k: f64)
                      -> Result<Self::CLRN, crate::co::error::Error>;

    /// Computes a [LRN][lrn] over the input Tensor `x`.
    /// [lrn]: https://en.wikipedia.org/wiki/lrnal_neural_network
    ///
    /// Saves the result to `result`.
    fn lrn(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>,
           config: &Self::CLRN) -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of a [LRN][lrn] over the input Tensor `x`.
    /// [lrn]: https://en.wikipedia.org/wiki/lrnal_neural_network
    ///
    /// Saves the result to `result_diff`.
    fn lrn_grad(&self,
                x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>,
                config: &Self::CLRN)
                -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for a Backend to support Pooling operations.
pub trait Pooling<F> : NN<F> {
    /// Creates a new PoolingConfig, which needs to be passed to further pooling Operations.
    fn new_pooling_config(&self, window: &[i32], stride: &[i32], padding: &[i32])
                          -> Result<Self::CPOOL, crate::co::error::Error>;

    /// Computes non-linear down-sampling ([max Pooling][pooling]) over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result`.
    fn pooling_max(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>,
                   config: &Self::CPOOL) -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of [max Pooling][pooling] over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result_diff`.
    fn pooling_max_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                        result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>,
                        config: &Self::CPOOL) -> Result<(), crate::co::error::Error>;


    /// Computes non-linear down-sampling ([average Pooling][pooling]) over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result`.
    fn pooling_avg(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>,
                   config: &Self::CPOOL) -> Result<(), crate::co::error::Error>;

    /// Computes the gradient of [average Pooling][pooling] over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result_diff`.
    fn pooling_avg_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                        result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>,
                        config: &Self::CPOOL) -> Result<(), crate::co::error::Error>;
}

/// Provides the functionality for a Backend to support Dropout operations.
pub trait Dropout<F> : NN<F> {
    /// Creates a new DropoutConfig, which needs to be passed to further dropout Operations.
    fn new_dropout_config(&self, dropout: f32, seed: u64)
                          -> Result<Self::CDROP, crate::co::error::Error>;

    /// Computes non-linear down-sampling ([max Pooling][pooling]) over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
    ///
    /// Saves the result to `result`.
    fn dropout(&self, x: &SharedTensor<F>, result: &mut SharedTensor<F>,
                   config: &Self::CDROP) -> Result<(), crate::co::error::Error>;

    /// Computes non-linear down-sampling ([max Pooling][pooling]) over the input Tensor `x`.
    /// [pooling]: https://en.wikipedia.org/wiki/Dropout_(neural_networks)
    ///
    /// Saves the result to `result`.
    fn dropout_grad(&self, x: &SharedTensor<F>, x_diff: &SharedTensor<F>,
                        result: &SharedTensor<F>, result_diff: &mut SharedTensor<F>,
                        config: &Self::CDROP) -> Result<(), crate::co::error::Error>;
}
