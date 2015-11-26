//! Provides a Rust wrapper around OpenCL's platform.

use super::api::types as cl;

#[derive(Debug, Copy, Clone)]
/// Defines a OpenCL Platform.
pub struct Platform {
    id: isize,
}

impl Platform {
    /// Initializes a new OpenCL platform.
    pub fn from_isize(id: isize) -> Platform {
        Platform { id: id }
    }

    /// Initializes a new OpenCL platform from its C type.
    pub fn from_c(id: cl::platform_id) -> Platform {
        Platform { id: id as isize }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::platform_id {
        self.id as cl::platform_id
    }
}
