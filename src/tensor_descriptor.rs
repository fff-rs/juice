//! Defines a Tensor Descriptor.
//!
//! A Tensor Descriptor is used to hold information about the data,
//! which is needed for the operations to obtain information about
//! the structure and dimensionality of the data.

use super::{API, Error};
use super::api::ffi::*;

#[derive(Debug, Clone)]
/// Describes a TensorDescriptor.
pub struct TensorDescriptor {
    id: isize,
}

impl Drop for TensorDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_tensor_descriptor(self.id_c());
    }
}

impl TensorDescriptor {
    /// Initializes a new CUDA cuDNN Tensor Descriptor.
    pub fn new(dims: &[i32], data_type: DataType) -> Result<TensorDescriptor, Error> {
        let nb_dims = dims.len() as i32;
        let dims_ptr = dims.as_ptr();
        let generic_tensor_desc = try!(API::create_tensor_descriptor());
        match data_type {
            DataType::Float => {
                let d_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
                try!(API::set_tensor_descriptor(generic_tensor_desc, d_type, nb_dims, dims_ptr, dims_ptr));
                Ok(TensorDescriptor::from_c(generic_tensor_desc))
            },
            DataType::Double => {
                let d_type = cudnnDataType_t::CUDNN_DATA_DOUBLE;
                try!(API::set_tensor_descriptor(generic_tensor_desc, d_type, nb_dims, dims_ptr, dims_ptr));
                Ok(TensorDescriptor::from_c(generic_tensor_desc))
            },
            DataType::Half => {
                let d_type = cudnnDataType_t::CUDNN_DATA_HALF;
                try!(API::set_tensor_descriptor(generic_tensor_desc, d_type, nb_dims, dims_ptr, dims_ptr));
                Ok(TensorDescriptor::from_c(generic_tensor_desc))
            }
        }
    }

    /// Initializes a new CUDA cuDNN Tensor Descriptor from its C type.
    pub fn from_c(id: cudnnTensorDescriptor_t) -> TensorDescriptor {
        TensorDescriptor { id: id as isize }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the CUDA cuDNN Tensor Descriptor as its C type.
    pub fn id_c(&self) -> cudnnTensorDescriptor_t {
        self.id as cudnnTensorDescriptor_t
    }
}

#[derive(Debug, Copy, Clone)]
/// Defines the available data types for the CUDA cuDNN data representation.
pub enum DataType {
    /// F32
    Float,
    /// F64
    Double,
    /// F16 (no native Rust support yet)
    Half,
}
