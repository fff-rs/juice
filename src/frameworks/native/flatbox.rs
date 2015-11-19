//! Provides a Box without any knowledge of its underlying type.
use memory::*;
use std::fmt;

/// A Box without any knowledge of its underlying type.
pub struct FlatBox {
    raw_box: *mut [u8]
}

impl FlatBox {
    /// Create FlatBox from Box, consuming it.
    pub fn from_box(b: Box<[u8]>) -> FlatBox {
        FlatBox {
            raw_box: Box::into_raw(b)
        }
    }
}

impl Drop for FlatBox {
    fn drop(&mut self) {
        unsafe {
            Box::from_raw(self.raw_box);
        }
    }
}

impl fmt::Debug for FlatBox {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "")
    }
}

impl IMemory for FlatBox {}
