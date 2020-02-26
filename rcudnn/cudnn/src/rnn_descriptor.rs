//! Defines a Recurrent Descriptor.
//!
//! A Recurrent Descriptor is used to hold information about the rnn,
//! which is needed for forward and backward rnnal operations.

use super::{API, Error};
use super::utils::DataType;
use ffi::*;
use ::{Cudnn, DropoutDescriptor};

use crate::cuda::CudaDeviceMemory;
use utils::DropoutConfig;

/// Describes a Recurrent Descriptor.
#[derive(Debug)]
pub struct RnnDescriptor {
    id: cudnnRNNDescriptor_t,
    dropout_config: DropoutConfig,
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
        handle: &Cudnn,
        hidden_size: i32,
        num_layers: i32,
        dropout_config: DropoutConfig,
        input_mode: cudnnRNNInputMode_t,
        direction: cudnnDirectionMode_t,
        mode: cudnnRNNMode_t,
        algorithm: cudnnRNNAlgo_t,
        data_type: DataType,
        padding_mode: cudnnRNNPaddingMode_t,
    ) -> Result<RnnDescriptor, Error> {
        let generic_rnn_desc = API::create_rnn_descriptor()?;
         API::set_rnn_descriptor(
             *handle.id_c(),
             generic_rnn_desc,
             hidden_size,
             num_layers,
             *dropout_config.dropout_desc().id_c(),
             input_mode,
             direction,
             mode,
             algorithm,
             data_type,
         )?;

        API::set_rnn_padding_mode(
            generic_rnn_desc,
            padding_mode,
        )?;

        Ok(RnnDescriptor {
            id: generic_rnn_desc,
            dropout_config,
        })
    }

    /// Initializes a new CUDA cuDNN RnnDescriptor from its C type.
    pub fn from_c(id: cudnnRNNDescriptor_t, dropout_config: DropoutConfig) -> RnnDescriptor {
        RnnDescriptor { id, dropout_config }
    }

    /// Returns the CUDA cuDNN RnnDescriptor as its C type.
    pub fn id_c(&self) -> &cudnnRNNDescriptor_t {
        &self.id
    }
}
