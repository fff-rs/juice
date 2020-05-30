//! Provides a binary on native CPU.

use crate::binary::IBinary;

#[derive(Debug, Default, Copy, Clone)]
/// Defines a host CPU binary.
pub struct Binary {
    id: isize,
}

impl Binary {
    /// Initializes the native CPU binary.
    pub fn new() -> Binary {
        Binary {
            id: 0,
        }
    }
}

impl IBinary for Binary {}
