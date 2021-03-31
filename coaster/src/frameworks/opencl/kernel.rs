//! Provides a Rust wrapper around OpenCL's Kernel.

use super::api::types as cl;
use operation::IOperation;

#[derive(Debug, Copy, Clone)]
/// Defines a OpenCL Kernel.
///
/// A Kernel is OpenCL's version of Coaster's [operation][operation].
/// [operation]: ../../operation/index.html
pub struct Kernel {
    id: isize,
}

impl Kernel {
    /// Initializes a new OpenCL device.
    pub fn from_isize(id: isize) -> Kernel {
        Kernel { id: id }
    }

    /// Initializes a new OpenCL device from its C type.
    pub fn from_c(id: cl::kernel_id) -> Kernel {
        Kernel { id: id as isize }
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::kernel_id {
        self.id as cl::kernel_id
    }
}

impl IOperation for Kernel {}
