//! Defines a Activation Descriptor.
//!
//! A Activation Descriptor is used to hold information about the rule,
//! which describes how to transform the data.

use super::{Error, API};
use crate::ffi::*;

#[derive(Debug, Clone)]
/// Describes a ActivationDescriptor.
pub struct ActivationDescriptor {
    id: cudnnActivationDescriptor_t,
}

impl Drop for ActivationDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_activation_descriptor(*self.id_c());
    }
}

impl ActivationDescriptor {
    /// Initializes a new CUDA cuDNN Activation Descriptor.
    pub fn new(mode: cudnnActivationMode_t) -> Result<ActivationDescriptor, Error> {
        let generic_activation_desc = API::create_activation_descriptor()?;
        API::set_activation_descriptor(
            generic_activation_desc,
            mode,
            cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN, // FIXME check if this makes sense
            ::std::f64::MAX,                                // FIXME make this public API
        )?;

        Ok(ActivationDescriptor::from_c(generic_activation_desc))
    }

    /// Initializes a new CUDA cuDNN Activation Descriptor from its C type.
    pub fn from_c(id: cudnnActivationDescriptor_t) -> ActivationDescriptor {
        ActivationDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN Activation Descriptor as its C type.
    pub fn id_c(&self) -> &cudnnActivationDescriptor_t {
        &self.id
    }
}
