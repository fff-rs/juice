//! Provides a binary on native CPU.

use binary::IBinary;
use frameworks::native::Function;

#[derive(Debug, Copy, Clone)]
/// Defines a host CPU binary.
pub struct Binary {
    id: isize,
    /// The initialized Blas Dot Operation.
    pub blas_dot: Function,
    /// The initialized Blas Scale Operation.
    pub blas_scale: Function,
    /// The initialized Blas Axpy Operation.
    pub blas_axpy: Function,
}

impl Binary {
    /// Initializes the native CPU binary.
    pub fn new() -> Binary {
        Binary {
            id: 0,
            blas_dot: Function::new(),
            blas_scale: Function::new(),
            blas_axpy: Function::new(),
        }
    }
}

impl IBinary for Binary {}
