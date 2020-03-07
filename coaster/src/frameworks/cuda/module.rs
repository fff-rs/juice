//! Provides a Rust wrapper around Cuda's Module.

use crate::binary::IBinary;

#[derive(Debug, Copy, Clone)]
/// Defines a Cuda Module.
///
/// A Module is Cuda's version of Coaster's [binary][binary].
/// [binary]: ../../binary/index.html
pub struct Module {
    id: isize,
}

impl Module {
    /// Initializes a new Cuda Module.
    pub fn from_isize(id: isize) -> Module {
        Module {
            id,
        }
    }

    // /// Initializes a new Cuda Module from its C type.
    // pub fn from_c(id: cl::kernel_id) -> Module {
    //     Module {
    //         id: id as isize,
    //         blas_dot: Function::from_isize(1),
    //         blas_scale: Function::from_isize(1),
    //         blas_axpy: Function::from_isize(1),
    //     }
    // }
    //
    // /// Returns the id as its C type.
    // pub fn id_c(&self) -> cl::kernel_id {
    //     self.id as cl::kernel_id
    // }
}

impl IBinary for Module {}
