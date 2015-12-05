//! Describes the high-level CUDA cuDNN instance.
//!
//! If you are using the high-level interface for cuDNN, you will start
//! by initilizing a new `Cudnn` instance. This initilizes the cuDNN resources,
//! stores the handle and manages future calls.

use super::{API, Error, TensorDescriptor, ScalParams};
use super::api::ffi::*;

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

    /// Computes the forward Sigmoid Activation function.
    ///
    /// Writes the result of the computation to `dest_data`.
    pub fn sigmoid_forward(
        &self,
        src_desc: TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams,
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
    pub fn sigmoid_backward(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams,
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
    pub fn relu_forward(
        &self,
        src_desc: TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams,
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
    pub fn relu_backward(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams,
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
    pub fn tanh_forward(
        &self,
        src_desc: TensorDescriptor,
        src_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
        dest_data: *mut ::libc::c_void,
        scale: ScalParams,
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
    pub fn tanh_backward(
        &self,
        src_desc: &TensorDescriptor,
        src_data: *const ::libc::c_void,
        src_diff_desc: &TensorDescriptor,
        src_diff_data: *const ::libc::c_void,
        dest_desc: TensorDescriptor,
        dest_data: *const ::libc::c_void,
        dest_diff_desc: TensorDescriptor,
        dest_diff_data: *mut ::libc::c_void,
        scale: ScalParams,
    ) -> Result<(), Error> {
        API::activation_backward(
            self.id_c(),
            cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
            scale.a, src_desc.id_c(), src_data, src_diff_desc.id_c(), src_diff_data,
            scale.b, dest_desc.id_c(), dest_data, dest_diff_desc.id_c(), dest_diff_data
        )
    }
}
