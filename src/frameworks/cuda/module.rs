//! Provides a Rust wrapper around Cuda's Module.

use binary::IBinary;
use super::function::Function;
use super::api::types as cl;
use super::api::API;

#[derive(Debug, Copy, Clone)]
/// Defines a Cuda Module.
///
/// A Module is Cuda's version of Collenchyma's [binary][binary].
/// [binary]: ../../binary/index.html
pub struct Module {
    id: isize,
    /// The initialized BLAS dot Operation.
    pub blas_dot: Function,
    /// The initialized BLAS scale Operation.
    pub blas_scale: Function,
    /// The initialized BLAS axpy Operation.
    pub blas_axpy: Function,
}

impl Module {
    /// Initializes a new OpenCL device.
    pub fn from_isize(id: isize) -> Module {
        Module {
            id: id,
            blas_dot: Function::from_isize(1),
            blas_scale: Function::from_isize(1),
            blas_axpy: Function::from_isize(1),
        }
    }

    /// Initializes a new OpenCL device from its C type.
    pub fn from_c(id: cl::kernel_id) -> Module {
        Module {
            id: id as isize,
            blas_dot: Function::from_isize(1),
            blas_scale: Function::from_isize(1),
            blas_axpy: Function::from_isize(1),
        }
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::kernel_id {
        self.id as cl::kernel_id
    }
}

impl IBinary for Module {}
