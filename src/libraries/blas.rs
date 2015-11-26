//! Provides backend-agnostic BLAS operations.
//!
//! BLAS (Basic Linear Algebra Subprograms) is a specification that prescribes a set of low-level
//! routines for performing common linear algebra operations such as vector addition, scalar
//! multiplication, dot products, linear combinations, and matrix multiplication. They are the de
//! facto standard low-level routines for linear algebra libraries; the routines have bindings for
//! both C and Fortran. Although the BLAS specification is general, BLAS implementations are often
//! optimized for speed on a particular machine, so using them can bring substantial performance
//! benefits. BLAS implementations will take advantage of special floating point hardware such as
//! vector registers or SIMD instructions.<br/>
//! [Source][blas-source]
//!
//! [blas-source]: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms

use framework::IFramework;
use binary::IBinary;

/// Provides the functionality for a backend to support Basic Linear Algebra Subprograms.
pub trait IBlas {
    /// The Binary representation for this Library.
    type B: IBlasBinary + IBinary;

    /// Level 1 operation
    fn dot(&self, a: i32) {
        // check if operation is provided;
        // create_mem;
        // sync_mem;
        self.binary().dot().compute(2)
    }

    /// Returns the binary representation
    fn binary(&self) -> Self::B;
}

/// Describes the operation binding for a Blas Binary implementation.
pub trait IBlasBinary {
    /// Describes the Dot Operation.
    type Dot: IOperationDot;

    /// Returns an initialized Dot operation.
    fn dot(&self) -> Self::Dot;
}

/// Describes a Dot Operation.
pub trait IOperationDot {
    /// Computes the Dot operation.
    fn compute(&self, a: i32);
}
