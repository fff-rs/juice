//! Provides the Transpose functionality for Matrix operations.

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

impl Transpose {
    /// Create a rust-blas `Transpose` from collenchyma-blas `Transpose`.
    pub fn to_rblas(&self) -> ::rblas::attribute::Transpose {
        match *self {
            Transpose::NoTrans => ::rblas::attribute::Transpose::NoTrans,
            Transpose::Trans => ::rblas::attribute::Transpose::Trans,
            Transpose::ConjTrans => ::rblas::attribute::Transpose::ConjTrans,
        }
    }
}
