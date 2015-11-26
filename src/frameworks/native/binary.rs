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
    /// Initializes a native CPU hardware.
    pub fn new(id: isize) -> Binary {
        Binary {
            id: id,
            blas_dot: Function::new(1)
        }
    }
}

impl IBinary for Binary {}
