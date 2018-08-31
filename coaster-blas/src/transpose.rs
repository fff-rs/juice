//! Provides the Transpose functionality for Matrix operations.
#[cfg(feature = "cuda")]
use std::convert::From;
#[cfg(feature = "cuda")]
use cublas::api::Operation;

#[derive(Debug, Copy, Clone)]
/// Possible transpose operations that can be applied in Level 2 and Level 3 BLAS operations.
pub enum Transpose {
    /// Take the matrix as it is.
    NoTrans,
    /// Take the transpose of the matrix.
    Trans,
    /// Take the conjugate transpose of the matrix.
    ConjTrans,
}

#[cfg(feature = "native")]
impl Transpose {
    /// Create a rust-blas `Transpose` from coaster-blas `Transpose`.
    pub fn to_rblas(&self) -> ::rblas::attribute::Transpose {
        match *self {
            Transpose::NoTrans => ::rblas::attribute::Transpose::NoTrans,
            Transpose::Trans => ::rblas::attribute::Transpose::Trans,
            Transpose::ConjTrans => ::rblas::attribute::Transpose::ConjTrans,
        }
    }
}

#[cfg(feature = "cuda")]
impl From<Operation> for Transpose {
    fn from(op: Operation) -> Self {
        match op {
            Operation::NoTrans => Transpose::NoTrans,
            Operation::Trans => Transpose::Trans,
            Operation::ConjTrans => Transpose::ConjTrans,
        }
    }
}

#[cfg(feature = "cuda")]
impl From<Transpose> for Operation {
    fn from(op: Transpose) -> Self {
        match op {
            Transpose::NoTrans => Operation::NoTrans,
            Transpose::Trans => Operation::Trans,
            Transpose::ConjTrans => Operation::ConjTrans,
        }
    }
}
