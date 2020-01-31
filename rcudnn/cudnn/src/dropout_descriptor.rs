//! Defines a Dropout Descriptor.
//!
//! A Tensor Descriptor is used to hold information about the probability
//! of dropping a value as well as an initial seed.

use super::{Error, API};
use crate::cudnn::Cudnn;
use crate::ffi::*;

use crate::cuda::CudaDeviceMemory;

#[derive(Debug, Clone)]
/// Describes a DropoutDescriptor.
pub struct DropoutDescriptor {
    id: cudnnDropoutDescriptor_t,
}

impl Drop for DropoutDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_dropout_descriptor(*self.id_c());
    }
}

impl DropoutDescriptor {
    /// Initializes a new CUDA cuDNN Dropout Descriptor.
    pub fn new(
        handle: &Cudnn,
        dropout: f32,
        seed: u64,
        reserve: &CudaDeviceMemory,
    ) -> Result<DropoutDescriptor, Error> {
        let generic_dropout_desc = API::create_dropout_descriptor()?;
        API::set_dropout_descriptor(
            generic_dropout_desc,
            *handle.id_c(),
            dropout,
            *reserve.id_c(),
            reserve.size(),
            seed,
        )?;

        Ok(DropoutDescriptor::from_c(generic_dropout_desc))
    }

    /// Get the size for a tensor
    pub fn get_required_size() -> usize {
        unimplemented!()
    }

    /// Initializes a new CUDA cuDNN Tensor Descriptor from its C type.
    pub fn from_c(id: cudnnDropoutDescriptor_t) -> DropoutDescriptor {
        DropoutDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN Tensor Descriptor as its C type.
    pub fn id_c(&self) -> &cudnnDropoutDescriptor_t {
        &self.id
    }
}
