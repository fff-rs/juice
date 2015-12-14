//! Describes the high-level CUDA cuDNN instance.
//!
//! If you are using the high-level interface for cuDNN, you will start
//! by initilizing a new `Cudnn` instance. This initilizes the cuDNN resources,
//! stores the handle and manages future calls.

use super::{API, Error, TensorDescriptor, FilterDescriptor, ConvolutionDescriptor};
use super::utils::{ConvolutionConfig, ScalParams};
use ffi::*;

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

    /// Initializes the parameters and configurations for running CUDA cuDNN operations.
    ///
    /// This includes finding the right convolution algorithm, workspace size and allocating
    /// that workspace.
    pub fn init_convolution(
        &self,
        mem_alloc: fn(usize) -> *mut ::libc::c_void,
        src_desc: cudnnTensorDescriptor_t,
        filter_desc: cudnnFilterDescriptor_t,
        conv_desc: cudnnConvolutionDescriptor_t,
        dest_desc: cudnnTensorDescriptor_t,
    ) -> Result<ConvolutionConfig, Error> {
        let algos_fwd = try!(API::find_convolution_forward_algorithm(self.id_c(), filter_desc, src_desc, conv_desc, dest_desc));
        let workspace_size_fwd = try!(API::get_convolution_forward_workspace_size(self.id_c(), algos_fwd[0].algo, filter_desc, src_desc, conv_desc, dest_desc));

        let algos_bwd = try!(API::find_convolution_backward_data_algorithm(self.id_c(), filter_desc, src_desc, conv_desc, dest_desc));
        let workspace_size_bwd = try!(API::get_convolution_backward_data_workspace_size(self.id_c(), algos_bwd[0].algo, filter_desc, src_desc, conv_desc, dest_desc));

        Ok(ConvolutionConfig::new(
            algos_fwd[0].algo, mem_alloc(workspace_size_fwd), workspace_size_fwd,
            algos_bwd[0].algo, mem_alloc(workspace_size_bwd), workspace_size_bwd,
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
        dest_desc: TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: TensorDescriptor,
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
        src_desc: TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
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
        dest_desc: TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: TensorDescriptor,
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
        src_desc: TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
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
        dest_desc: TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: TensorDescriptor,
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
        conv_desc: &ConvolutionDescriptor,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        filter_desc: &FilterDescriptor,
        filter_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::convolution_forward(
            self.id_c(),
            *conv_config.forward_algo(), conv_desc.id_c(), *conv_config.forward_workspace(), *conv_config.forward_workspace_size(),
            scale.a, src_desc.id_c(), src_data, filter_desc.id_c(), filter_data,
            scale.b, dest_desc.id_c(), dest_data
        )
    }

    /// Computes the backward Convolution function w.r.t the data.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn convolution_backward<T>(
        &self,
        conv_config: &ConvolutionConfig,
        conv_desc: &ConvolutionDescriptor,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        filter_desc: &FilterDescriptor,
        filter_data: *const ::libc::c_void,
        dest_grad_desc: TensorDescriptor,
        dest_grad_data: *mut ::libc::c_void,
        scale: ScalParams<T>,
    ) -> Result<(), Error> {
        API::convolution_backward_data(
            self.id_c(),
            *conv_config.backward_algo(), conv_desc.id_c(), *conv_config.backward_workspace(), *conv_config.backward_workspace_size(),
            scale.a, src_diff_desc.id_c(), src_diff_data, filter_desc.id_c(), filter_data,
            scale.b, dest_grad_desc.id_c(), dest_grad_data
        )
    }
}
