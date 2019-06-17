//! Defines a Tensor Descriptor.
//!
//! A Tensor Descriptor is used to hold information about the data,
//! which is needed for the operations to obtain information about
//! the structure and dimensionality of the data.

use super::{API, Error};
use super::utils::DataType;
use ffi::*;

#[derive(Debug, Clone)]
/// Describes a TensorDescriptor.
pub struct TensorDescriptor {
    id: cudnnTensorDescriptor_t,
}

impl Drop for TensorDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_tensor_descriptor(*self.id_c());
    }
}

impl TensorDescriptor {
    /// Initializes a new CUDA cuDNN Tensor Descriptor.
    pub fn new(dims: &[i32], strides: &[i32], data_type: DataType) -> Result<TensorDescriptor, Error> {

        let nb_dims = dims.len() as i32;
        if nb_dims < 3 { return Err(Error::BadParam("CUDA cuDNN only supports Tensors with 3 to 8 dimensions.")) }

        let dims_ptr = dims.as_ptr();
        let strides_ptr = strides.as_ptr();
        let generic_tensor_desc = API::create_tensor_descriptor()?;
        match data_type {
            DataType::Float => {
                let d_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
                API::set_tensor_descriptor(generic_tensor_desc, d_type, nb_dims, dims_ptr, strides_ptr)?;
                Ok(TensorDescriptor::from_c(generic_tensor_desc))
            },
            DataType::Double => {
                let d_type = cudnnDataType_t::CUDNN_DATA_DOUBLE;
                API::set_tensor_descriptor(generic_tensor_desc, d_type, nb_dims, dims_ptr, strides_ptr)?;
                Ok(TensorDescriptor::from_c(generic_tensor_desc))
            },
            DataType::Half => {
                let d_type = cudnnDataType_t::CUDNN_DATA_HALF;
                API::set_tensor_descriptor(generic_tensor_desc, d_type, nb_dims, dims_ptr, strides_ptr)?;
                Ok(TensorDescriptor::from_c(generic_tensor_desc))
            }
        }
    }

    /// Initializes a new CUDA cuDNN Tensor Descriptor from its C type.
    pub fn from_c(id: cudnnTensorDescriptor_t) -> TensorDescriptor {
        TensorDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN Tensor Descriptor as its C type.
    pub fn id_c(&self) -> &cudnnTensorDescriptor_t {
        &self.id
    }
}
