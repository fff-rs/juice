//! Defines a Recurrent Descriptor.
//!
//! A Recurrent Descriptor is used to hold information about the rnn,
//! which is needed for forward and backward rnnal operations.

use super::{API, Error};
use super::utils::DataType;
use ffi::*;

#[derive(Debug, Clone)]
/// Describes a Recurrent Descriptor.
pub struct RnnDescriptor {
    id: cudnnRnnDescriptor_t,
}

impl Drop for RnnDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_rnn_descriptor(*self.id_c());
    }
}

impl RnnDescriptor {
    /// Initializes a new CUDA cuDNN RnnDescriptor.
    pub fn new(pad: &[i32], filter_stride: &[i32], data_type: DataType) -> Result<RnnDescriptor, Error> {
        let array_length = pad.len() as i32;
        let upscale: Vec<i32> = ::std::iter::repeat(1i32).take(array_length as usize).collect();

        let generic_rnn_desc = try!(API::create_rnn_descriptor());
        let data_type = match data_type {
            DataType::Float => cudnnDataType_t::CUDNN_DATA_FLOAT,
            DataType::Double => cudnnDataType_t::CUDNN_DATA_DOUBLE),
            DataType::Half => cudnnDataType_t::CUDNN_DATA_HALF),
            _ => return Err(Error::InvalidValue("Invalid data type value passed")),
        }
        API::set_rnn_descriptor(generic_rnn_desc, );
        Ok(RnnDescriptor::from_c(generic_rnn_desc))

    }

    /// Initializes a new CUDA cuDNN RnnDescriptor from its C type.
    pub fn from_c(id: cudnnRnnDescriptor_t) -> RnnDescriptor {
        RnnDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN RnnDescriptor as its C type.
    pub fn id_c(&self) -> &cudnnRnnDescriptor_t {
        &self.id
    }
}
