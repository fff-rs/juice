//! Defines a LRN Descriptor.

use super::{API, Error};
use ffi::*;

#[derive(Debug, Clone)]
/// Describes a LRN Descriptor.
pub struct NormalizationDescriptor {
    id: cudnnLRNDescriptor_t,
}

impl Drop for NormalizationDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_lrn_descriptor(*self.id_c());
    }
}

impl NormalizationDescriptor {
    /// Initializes a new CUDA cuDNN LRNDescriptor.
    pub fn new(lrn_n: u32, lrn_alpha: f64, lrn_beta: f64, lrn_k: f64) -> Result<NormalizationDescriptor, Error> {
        let generic_lrn_desc = API::create_lrn_descriptor()?;
        API::set_lrn_descriptor(generic_lrn_desc, lrn_n, lrn_alpha, lrn_beta, lrn_k)?;
        Ok(NormalizationDescriptor::from_c(generic_lrn_desc))
    }

    /// Initializes a new CUDA cuDNN NormalizationDescriptor from its C type.
    pub fn from_c(id: cudnnLRNDescriptor_t) -> NormalizationDescriptor {
        NormalizationDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN NormalizationDescriptor as its C type.
    pub fn id_c(&self) -> &cudnnLRNDescriptor_t {
        &self.id
    }
}
