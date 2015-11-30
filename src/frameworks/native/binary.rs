//! Provides a binary on native CPU.

use binary::IBinary;
use frameworks::native::Function;

#[derive(Debug, Copy, Clone)]
/// Defines a host CPU binary.
pub struct Binary {
    id: isize,
    /// The initialized Blas Asum Operation.
    pub blas_asum: Function,
    /// The initialized Blas Axpy Operation.
    pub blas_axpy: Function,
    /// The initialized Blas Copy Operation.
    pub blas_copy: Function,
    /// The initialized Blas Dot Operation.
    pub blas_dot: Function,
    /// The initialized Blas Nrm2 Operation.
    pub blas_nrm2: Function,
    /// The initialized Blas Scale Operation.
    pub blas_scale: Function,
    /// The initialized Blas Swap Operation.
    pub blas_swap: Function,
}

impl Binary {
    /// Initializes the native CPU binary.
    pub fn new() -> Binary {
        Binary {
            id: 0,
            blas_asum: Function::new(),
            blas_axpy: Function::new(),
            blas_copy: Function::new(),
            blas_dot: Function::new(),
            blas_nrm2: Function::new(),
            blas_scale: Function::new(),
            blas_swap: Function::new(),
        }
    }
}

impl IBinary for Binary {}
