//! Provides a Rust wrapper around OpenCL's Program.

use binary::IBinary;
use super::api::types as cl;

#[derive(Debug, Copy, Clone)]
/// Defines a OpenCL Program.
///
/// A Program is OpenCL's version of Collenchyma's [binary][binary].
/// [binary]: ../../binary/index.html
pub struct Program {
    id: isize,
}

impl Program {
    /// Initializes a new OpenCL device.
    pub fn from_isize(id: isize) -> Program {
        Program {
            id: id,
        }
    }

    /// Initializes a new OpenCL device from its C type.
    pub fn from_c(id: cl::kernel_id) -> Program {
        Program {
            id: id as isize,
        }
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::kernel_id {
        self.id as cl::kernel_id
    }
}

impl IBinary for Program {}
