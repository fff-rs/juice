//! Describes the high-level CUDA cuDNN instance.
//!
//! If you are using the high-level interface for cuDNN, you will start
//! by initilizing a new `Cudnn` instance. This initilizes the cuDNN resources,
//! stores the handle and manages future calls.

use super::*;
use super::utils::{ConvolutionConfig, DataTypeInfo,
                   NormalizationConfig, PoolingConfig, ScalParams};
use num::traits::Float;
use std::mem::transmute_copy;

#[derive(Debug, Clone)]
/// Provides a the high-level interface to CUDA's cuDNN.
pub struct Cudnn {
    id: cudnnHandle_t,
}

unsafe impl ::std::marker::Sync for Cudnn {}

impl Drop for Cudnn {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy(*self.id_c());
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
        Cudnn { id: id }
    }

    /// Returns the CUDA cuDNN Context as its C type.
    pub fn id_c(&self) -> &cudnnHandle_t {
        &self.id
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
        src_desc: &TensorDescriptor,
        conv_desc: ConvolutionDescriptor,
        filter_desc: FilterDescriptor,
        dest_desc: &TensorDescriptor,
    ) -> Result<ConvolutionConfig, Error> {
        let algos_fwd = try!(API::find_convolution_forward_algorithm(*self.id_c(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()));
        let workspace_size_fwd = try!(API::get_convolution_forward_workspace_size(*self.id_c(), algos_fwd[0].algo, *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()));

        let algos_filter_bwd = try!(API::find_convolution_backward_filter_algorithm(*self.id_c(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()));
        let workspace_filter_size_bwd = try!(API::get_convolution_backward_filter_workspace_size(*self.id_c(), algos_filter_bwd[0].algo, *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()));

        let algos_data_bwd = try!(API::find_convolution_backward_data_algorithm(*self.id_c(), *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()));
        let workspace_data_size_bwd = try!(API::get_convolution_backward_data_workspace_size(*self.id_c(), algos_data_bwd[0].algo, *filter_desc.id_c(), *conv_desc.id_c(), *src_desc.id_c(), *dest_desc.id_c()));

        Ok(
            ConvolutionConfig::new(
                algos_fwd[0].algo, workspace_size_fwd,
                algos_filter_bwd[0].algo, workspace_filter_size_bwd,
                algos_data_bwd[0].algo, workspace_data_size_bwd,
                conv_desc, filter_desc
            )
        )
    }

    /// Initializes the parameters and configurations for running CUDA cuDNN LRN operations.
    pub fn init_normalization(
        &self,
        lrn_n: u32,
        lrn_alpha: f64,
        lrn_beta: f64,
        lrn_k: f64
    ) -> Result<NormalizationConfig, Error> {
        let norm_desc = try!(NormalizationDescriptor::new(lrn_n, lrn_alpha, lrn_beta, lrn_k));
        Ok(NormalizationConfig::new(norm_desc))
    }

    /// Initializes the parameters and configurations for running CUDA cuDNN Pooling operations.
    pub fn init_pooling(
        &self,
        window: &[i32],
        padding: &[i32],
        stride: &[i32],
    ) -> Result<PoolingConfig, Error> {
        let avg = try!(PoolingDescriptor::new(cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, window, padding, stride));
        let max = try!(PoolingDescriptor::new(cudnnPoolingMode_t::CUDNN_POOLING_MAX, window, padding, stride));
        Ok(PoolingConfig::new(avg, max))
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::activation_forward(
            *self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::activation_backward(
            *self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) },
            *dest_desc.id_c(), dest_data, *dest_diff_desc.id_c(), dest_diff_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::activation_forward(
            *self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::activation_backward(
            *self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) },
            *dest_desc.id_c(), dest_data, *dest_diff_desc.id_c(), dest_diff_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::activation_forward(
            *self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::activation_backward(
            *self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) },
            *dest_desc.id_c(), dest_data, *dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward Convolution function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn convolution_forward<T>(
        &self,
        conv_config: &ConvolutionConfig,
        workspace: *mut ::libc::c_void,
        filter_data: *const ::libc::c_void,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::convolution_forward(
            *self.id_c(),
            *conv_config.forward_algo(), *conv_config.conv_desc().id_c(), workspace, *conv_config.forward_workspace_size(),
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *conv_config.filter_desc().id_c(), filter_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward Convolution function w.r.t the bias.
    ///
    /// Writes the result of the computation to `bias_grad_data`.
    pub fn convolution_backward_bias<T>(
        &self,
        dest_grad_desc: &TensorDescriptor,
        dest_grad_data: *const ::libc::c_void,
        bias_grad_desc: &TensorDescriptor,
        bias_grad_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::convolution_backward_bias(
            *self.id_c(),
            unsafe { transmute_copy(&&scale.a) },
            *dest_grad_desc.id_c(), dest_grad_data,
            unsafe { transmute_copy(&&scale.b) }, *bias_grad_desc.id_c(), bias_grad_data
        )
    }

    /// Computes the backward Convolution function w.r.t the filter.
    ///
    /// Writes the result of the computation to `filter_data`.
    pub fn convolution_backward_filter<T>(
        &self,
        conv_config: &ConvolutionConfig,
        workspace: *mut ::libc::c_void,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_grad_desc: &TensorDescriptor,
        dest_grad_data: *const ::libc::c_void,
        filter_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::convolution_backward_filter(
            *self.id_c(),
            *conv_config.backward_filter_algo(), *conv_config.conv_desc().id_c(), workspace, *conv_config.backward_filter_workspace_size(),
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *dest_grad_desc.id_c(), dest_grad_data,
            unsafe { transmute_copy(&&scale.b) }, *conv_config.filter_desc().id_c(), filter_data
        )
    }

    /// Computes the backward Convolution function w.r.t the data.
    ///
    /// Writes the result of the computation to `src_grad_data`.
    pub fn convolution_backward_data<T>(
        &self,
        conv_config: &ConvolutionConfig,
        workspace: *mut ::libc::c_void,
        filter_data: *const ::libc::c_void,
        dest_grad_desc: &TensorDescriptor,
        dest_grad_data: *const ::libc::c_void,
        src_grad_desc: &TensorDescriptor,
        src_grad_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::convolution_backward_data(
            *self.id_c(),
            *conv_config.backward_data_algo(), *conv_config.conv_desc().id_c(), workspace, *conv_config.backward_data_workspace_size(),
            unsafe { transmute_copy(&&scale.a) },
            *conv_config.filter_desc().id_c(), filter_data, *dest_grad_desc.id_c(), dest_grad_data,
            unsafe { transmute_copy(&&scale.b) }, *src_grad_desc.id_c(), src_grad_data
        )
    }

    /// Computes the forward softmax function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn softmax_forward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::softmax_forward(
            *self.id_c(), cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward softmax function.
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::softmax_backward(
            *self.id_c(), cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_diff_desc.id_c(), dest_diff_data
        )
    }

    /// Computes the forward logarithmic softmax function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn log_softmax_forward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: &TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::softmax_forward(
            *self.id_c(), cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward logarithmic softmax function.
    ///
    /// Writes the result of the computation to `dest_diff_data`.
    pub fn log_softmax_backward<T>(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_diff_desc: &TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::softmax_backward(
            *self.id_c(), cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_diff_desc.id_c(), dest_diff_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::lrn_cross_channel_forward(
            *self.id_c(), *normalization_conf.lrn_desc().id_c(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::lrn_cross_channel_backward(
            *self.id_c(), *normalization_conf.lrn_desc().id_c(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) },
            *dest_desc.id_c(), dest_data, *dest_diff_desc.id_c(), dest_diff_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::pooling_forward(
            *self.id_c(), *pooling_conf.pooling_avg_desc().id_c(),
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::pooling_backward(
            *self.id_c(), *pooling_conf.pooling_avg_desc().id_c(),
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) },
            *dest_desc.id_c(), dest_data, *dest_diff_desc.id_c(), dest_diff_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::pooling_forward(
            *self.id_c(), *pooling_conf.pooling_max_desc().id_c(),
            unsafe { transmute_copy(&&scale.a) }, *src_desc.id_c(), src_data,
            unsafe { transmute_copy(&&scale.b) }, *dest_desc.id_c(), dest_data
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
    ) -> Result<(), Error>
        where T: Float + DataTypeInfo,
    {
        API::pooling_backward(
            *self.id_c(), *pooling_conf.pooling_max_desc().id_c(),
            unsafe { transmute_copy(&&scale.a) },
            *src_desc.id_c(), src_data, *src_diff_desc.id_c(), src_diff_data,
            unsafe { transmute_copy(&&scale.b) },
            *dest_desc.id_c(), dest_data, *dest_diff_desc.id_c(), dest_diff_data
        )
    }
}
