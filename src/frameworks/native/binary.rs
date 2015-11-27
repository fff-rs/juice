//! Provides a binary on native CPU.

use binary::IBinary;
use frameworks::native::Function;

#[derive(Debug, Copy, Clone)]
/// Defines a host CPU binary.
pub struct Binary {
    id: isize,
    /// The initialized Blas Dot Operation.
    pub blas_dot: Function,
}

impl Binary {
    /// Initializes the native CPU binary.
    pub fn new() -> Binary {
        Binary {
            id: 0,
            blas_dot: Function::new()
        }
    }
}

impl IBinary for Binary {}
