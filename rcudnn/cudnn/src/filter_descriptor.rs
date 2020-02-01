//! Defines a Filter Descriptor.
//!
//! A Filter Descriptor is used to hold information about the Filter,
//! which is needed for forward and backward convolutional operations.

use super::utils::DataType;
use super::{Error, API};
use crate::ffi::*;

#[derive(Debug, Clone)]
/// Describes a Filter Descriptor.
pub struct FilterDescriptor {
    id: cudnnFilterDescriptor_t,
}

impl Drop for FilterDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_filter_descriptor(*self.id_c());
    }
}

impl FilterDescriptor {
    /// Initializes a new CUDA cuDNN FilterDescriptor.
    pub fn new(filter_dim: &[i32], data_type: DataType) -> Result<FilterDescriptor, Error> {
        let nb_dims = filter_dim.len() as i32;
        let tensor_format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        let generic_filter_desc = API::create_filter_descriptor()?;
        match data_type {
            DataType::Float => {
                let d_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
                API::set_filter_descriptor(
                    generic_filter_desc,
                    d_type,
                    tensor_format,
                    nb_dims,
                    filter_dim.as_ptr(),
                )?;
                Ok(FilterDescriptor::from_c(generic_filter_desc))
            }
            DataType::Double => {
                let d_type = cudnnDataType_t::CUDNN_DATA_DOUBLE;
                API::set_filter_descriptor(
                    generic_filter_desc,
                    d_type,
                    tensor_format,
                    nb_dims,
                    filter_dim.as_ptr(),
                )?;
                Ok(FilterDescriptor::from_c(generic_filter_desc))
            }
            DataType::Half => {
                let d_type = cudnnDataType_t::CUDNN_DATA_HALF;
                API::set_filter_descriptor(
                    generic_filter_desc,
                    d_type,
                    tensor_format,
                    nb_dims,
                    filter_dim.as_ptr(),
                )?;
                Ok(FilterDescriptor::from_c(generic_filter_desc))
            }
        }
    }

    /// Initializes a new CUDA cuDNN FilterDescriptor from its C type.
    pub fn from_c(id: cudnnFilterDescriptor_t) -> FilterDescriptor {
        FilterDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN FilterDescriptor as its C type.
    pub fn id_c(&self) -> &cudnnFilterDescriptor_t {
        &self.id
    }
}
