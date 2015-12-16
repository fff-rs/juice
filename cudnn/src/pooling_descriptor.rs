//! Defines a Pooling Descriptor.

use super::{API, Error};
use ffi::*;

#[derive(Debug, Clone)]
/// Describes a Pooling Descriptor.
pub struct PoolingDescriptor {
    id: isize,
}

impl Drop for PoolingDescriptor {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        API::destroy_pooling_descriptor(self.id_c());
    }
}

impl PoolingDescriptor {
    /// Initializes a new CUDA cuDNN Pooling Descriptor.
    pub fn new(mode: cudnnPoolingMode_t, window: &[i32], padding: &[i32], stride: &[i32]) -> Result<PoolingDescriptor, Error> {
        let nb_dims: i32 = window.len() as i32;
        let generic_pooling_desc = try!(API::create_pooling_descriptor());
        try!(API::set_pooling_descriptor(generic_pooling_desc, mode, nb_dims, window.as_ptr(), padding.as_ptr(), stride.as_ptr()));
        Ok(PoolingDescriptor::from_c(generic_pooling_desc))
    }

    /// Initializes a new CUDA cuDNN PoolingDescriptor from its C type.
    pub fn from_c(id: cudnnPoolingDescriptor_t) -> PoolingDescriptor {
        PoolingDescriptor { id: id as isize }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the CUDA cuDNN Pooling Descriptor as its C type.
    pub fn id_c(&self) -> cudnnPoolingDescriptor_t {
        self.id as cudnnPoolingDescriptor_t
    }
}
