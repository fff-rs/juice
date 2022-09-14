//! Provides a Box without any knowledge of its underlying type.

use crate::device::IMemory;
use std::fmt;
use std::mem;
use std::slice;

/// A Box without any knowledge of its underlying type.
pub struct FlatBox {
    len: usize,
    raw_box: *mut [u8],
}

impl FlatBox {
    /// Create FlatBox from Box, consuming it.
    pub fn from_box(b: Box<[u8]>) -> FlatBox {
        FlatBox {
            len: b.len(),
            raw_box: Box::into_raw(b),
        }
    }

    /// Access memory as slice.
    ///
    /// The preffered way to access native memory.
    pub fn as_slice<T>(&self) -> &[T] {
        unsafe { slice::from_raw_parts_mut(self.raw_box as *mut T, self.len / mem::size_of::<T>()) }
    }

    /// Access memory as mutable slice.
    ///
    /// The preffered way to access native memory.
    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.raw_box as *mut T, self.len / mem::size_of::<T>()) }
    }

    /// Returns memory size of the Flatbox.
    pub fn byte_size(&self) -> usize {
        self.len
    }
}

impl Drop for FlatBox {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.raw_box));
        }
    }
}

impl fmt::Debug for FlatBox {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FlatBox of length {}", &self.len)
    }
}

impl IMemory for FlatBox {}
