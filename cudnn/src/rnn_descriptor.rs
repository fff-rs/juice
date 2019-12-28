//! Defines a Recurrent Descriptor.
//!
//! A Recurrent Descriptor is used to hold information about the rnn,
//! which is needed for forward and backward rnnal operations.

use super::{API, Error};
use super::utils::DataType;
use ffi::*;
use ::{Cudnn, DropoutDescriptor};

#[derive(Debug, Clone)]
/// Describes a Recurrent Descriptor.
pub struct RnnDescriptor {
    id: cudnnRNNDescriptor_t,
}

impl Drop for RnnDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_rnn_descriptor(*self.id_c());
    }
}

impl RnnDescriptor {
    /// Initializes a new CUDA cuDNN RnnDescriptor.
    pub fn new(
        handle: cudnnHandle_t,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: &DropoutDescriptor,
        input_mode: cudnnRNNInputMode_t,
        direction: cudnnDirectionMode_t,
        mode: cudnnRNNMode_t,
        algorithm: cudnnRNNAlgo_t,
        data_type: DataType,
    ) -> Result<RnnDescriptor, Error> {
        let generic_rnn_desc = API::create_rnn_descriptor()?;
         API::set_rnn_descriptor(
            handle,
            generic_rnn_desc,
            hidden_size,
            num_layers,
            *dropout_desc.id_c(),
            input_mode,
            direction,
            mode,
            algorithm,
            data_type,
        )?;

        Ok(RnnDescriptor {
            id: generic_rnn_desc
        })
    }

    /// Initializes a new CUDA cuDNN RnnDescriptor from its C type.
    pub fn from_c(id: cudnnRNNDescriptor_t) -> RnnDescriptor {
        RnnDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN RnnDescriptor as its C type.
    pub fn id_c(&self) -> &cudnnRNNDescriptor_t {
        &self.id
    }
}
