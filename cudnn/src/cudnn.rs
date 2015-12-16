//! Describes the high-level CUDA cuDNN instance.
//!
//! If you are using the high-level interface for cuDNN, you will start
//! by initilizing a new `Cudnn` instance. This initilizes the cuDNN resources,
//! stores the handle and manages future calls.

use super::*;
use super::utils::{ConvolutionConfig, NormalizationConfig, PoolingConfig, ScalParams};

#[derive(Debug, Clone)]
/// Provides a the high-level interface to CUDA's cuDNN.
pub struct Cudnn {
    id: isize,
}

impl Drop for Cudnn {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy(self.id_c());
    }
}

impl Cudnn {
    /// Initializes a new CUDA cuDNN context.
    ///
    /// Make sure your current CUDA device is cuDNN enabled.
    pub fn new() -> Result<Cudnn, Error> {
        Ok(Cudnn::from_c(try!(API::init())))
    }

    /// Initializes a new CUDA cuDNN Context from its C type.
    pub fn from_c(id: cudnnHandle_t) -> Cudnn {
        Cudnn { id: id as isize }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the CUDA cuDNN Context as its C type.
    pub fn id_c(&self) -> cudnnHandle_t {
        self.id as cudnnHandle_t
    }

    /// Returns the version of the CUDA cuDNN library.
    pub fn version() -> usize {
        API::get_version()
    }

    /// Initializes the parameters and configurations for running CUDA cuDNN convolution operations.
    ///
    /// This includes finding the right convolution algorithm, workspace size and allocating
    /// that workspace.
    pub fn init_convolution(
        &self,
        mem_alloc: fn(usize) -> *mut ::libc::c_void,
        filter_data: *const ::libc::c_void,
        src_desc: &TensorDescriptor,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvolutionConfig, Error> {
        let algos_fwd = try!(API::find_convolution_forward_algorithm(self.id_c(), filter_desc.id_c(), src_desc.id_c(), conv_desc.id_c(), dest_desc.id_c()));
        let workspace_size_fwd = try!(API::get_convolution_forward_workspace_size(self.id_c(), algos_fwd[0].algo, filter_desc.id_c(), src_desc.id_c(), conv_desc.id_c(), dest_desc.id_c()));

        let algos_bwd = try!(API::find_convolution_backward_data_algorithm(self.id_c(), filter_desc.id_c(), src_desc.id_c(), conv_desc.id_c(), dest_desc.id_c()));
        let workspace_size_bwd = try!(API::get_convolution_backward_data_workspace_size(self.id_c(), algos_bwd[0].algo, filter_desc.id_c(), src_desc.id_c(), conv_desc.id_c(), dest_desc.id_c()));

        Ok(
            ConvolutionConfig::new(
                algos_fwd[0].algo, mem_alloc(workspace_size_fwd), workspace_size_fwd,
                algos_bwd[0].algo, mem_alloc(workspace_size_bwd), workspace_size_bwd,
                conv_desc.id_c(), filter_desc.id_c(), filter_data
            )
        )
    }

    /// Initializes the parameters and configurations for running CUDA cuDNN LRN operations.
    pub fn init_normalization(
        &self,
        lrn_desc: &NormalizationDescriptor,
    ) -> Result<NormalizationConfig, Error> {
        Ok(NormalizationConfig::new(lrn_desc.id_c()))
    }

    /// Initializes the parameters and configurations for running CUDA cuDNN Pooling operations.
    pub fn init_pooling(
        &self,
        window: &[i32],
        padding: &[i32],
        stride: &[i32],
    ) -> Result<PoolingConfig, Error> {
        Ok(PoolingConfig::new(
            try!(PoolingDescriptor::new(cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, window, padding, stride)).id_c(),
            try!(PoolingDescriptor::new(cudnnPoolingMode_t::CUDNN_POOLING_MAX, window, padding, stride)).id_c(),
        ))
    }

    /// Computes the forward Sigmoid Activation function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn sigmoid_forward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::activation_forward(
            self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
            scale.a, src_desc.id_c(), src_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward Sigmoid Activation function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn sigmoid_backward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::activation_backward(
            self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_desc.id_c(), dest_data, dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward Rectified Linear Activation function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn relu_forward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::activation_forward(
            self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            scale.a, src_desc.id_c(), src_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward Rectified Linear Activation function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn relu_backward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::activation_backward(
            self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_desc.id_c(), dest_data, dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward Hyperbolic Tangent Activation function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn tanh_forward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::activation_forward(
            self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
            scale.a, src_desc.id_c(), src_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward Hyperbolic Tangent Activation function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn tanh_backward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::activation_backward(
            self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_desc.id_c(), dest_data, dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward Convolution function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn convolution_forward<T>(
        &self,
        conv_config: &ConvolutionConfig,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::convolution_forward(
            self.id_c(),
            *conv_config.forward_algo(), *conv_config.conv_desc(), *conv_config.forward_workspace(), *conv_config.forward_workspace_size(),
            scale.a, src_desc.id_c(), src_data, *conv_config.filter_desc(), *conv_config.filter_data(),
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward Convolution function w.r.t the data.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn convolution_backward<T>(
        &self,
        conv_config: &ConvolutionConfig,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_grad_desc: &TensorDescriptor,
        dest_grad_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::convolution_backward_data(
            self.id_c(),
            *conv_config.backward_algo(), *conv_config.conv_desc(), *conv_config.backward_workspace(), *conv_config.backward_workspace_size(),
            scale.a, src_diff_desc.id_c(), src_diff_data, *conv_config.filter_desc(), *conv_config.filter_data(),
            scale.b, dest_grad_desc.id_c(), dest_grad_data
        )
    }

    /// Computes the forward softmax activation function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn softmax_forward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::softmax_forward(
            self.id_c(), cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            scale.a, src_desc.id_c(), src_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward softmax activation function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn softmax_backward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::softmax_backward(
            self.id_c(), cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward local response normalization function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn lrn_forward<T>(
        &self,
        normalization_conf: &NormalizationConfig,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::lrn_cross_channel_forward(
            self.id_c(), *normalization_conf.lrn_desc(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
            scale.a, src_desc.id_c(), src_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward local response normalization function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn lrn_backward<T>(
        &self,
        normalization_conf: &NormalizationConfig,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::lrn_cross_channel_backward(
            self.id_c(), *normalization_conf.lrn_desc(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_desc.id_c(), dest_data, dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward average pooling function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn pooling_avg_forward<T>(
        &self,
        pooling_conf: &PoolingConfig,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::pooling_forward(
            self.id_c(), *pooling_conf.pooling_avg_desc(),
            scale.a, src_desc.id_c(), src_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward average pooling function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn pooling_avg_backward<T>(
        &self,
        pooling_conf: &PoolingConfig,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::pooling_backward(
            self.id_c(), *pooling_conf.pooling_avg_desc(),
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_desc.id_c(), dest_data, dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward max pooling function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn pooling_max_forward<T>(
        &self,
        pooling_conf: &PoolingConfig,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::pooling_forward(
            self.id_c(), *pooling_conf.pooling_max_desc(),
            scale.a, src_desc.id_c(), src_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward max pooling function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn pooling_max_backward<T>(
        &self,
        pooling_conf: &PoolingConfig,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::pooling_backward(
            self.id_c(), *pooling_conf.pooling_max_desc(),
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_desc.id_c(), dest_data, dest_diff_desc.id_c(), dest_diff_data
        )
    }
}
