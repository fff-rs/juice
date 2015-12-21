//! Defines a Pooling Descriptor.

use super::{API, Error};
use ffi::*;

#[derive(Debug, Clone)]
/// Describes a Pooling Descriptor.
pub struct PoolingDescriptor {
    id: cudnnPoolingDescriptor_t,
}

impl Drop for PoolingDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_pooling_descriptor(*self.id_c()).unwrap();
    }
}

impl PoolingDescriptor {
    /// Initializes a new CUDA cuDNN Pooling Descriptor.
    pub fn new(mode: cudnnPoolingMode_t, window: &[i32], padding: &[i32], stride: &[i32]) -> Result<PoolingDescriptor, Error> {
        let generic_pooling_desc = try!(API::create_pooling_descriptor());
        try!(API::set_pooling_descriptor(generic_pooling_desc, mode, window.len() as i32, window.as_ptr(), padding.as_ptr(), stride.as_ptr()));
        Ok(PoolingDescriptor::from_c(generic_pooling_desc))
    }

    /// Initializes a new CUDA cuDNN PoolingDescriptor from its C type.
    pub fn from_c(id: cudnnPoolingDescriptor_t) -> PoolingDescriptor {
        PoolingDescriptor { id: id }
    }

    /// Returns the CUDA cuDNN Pooling Descriptor as its C type.
    pub fn id_c(&self) -> &cudnnPoolingDescriptor_t {
        &self.id
    }
}
