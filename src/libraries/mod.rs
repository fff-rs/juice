//! Exposes the explicit implementations of [operations][operation] on a [backend][backend].
//! [operation]: ../operation/index.html
//! [backend]: ../backend/index.html
//!
//! A library can be directly implemented for a backend. A library provides 'provided methods',
//! meaning, that no code needs to be writen at the implementation of the library on a backend.
//!
//! A library plays a significant role in Collenchyma and allows for the native interface for
//! executing operations, which is no-different than calling a 'normal' Rust method. The library
//! declares the functionality and automatically manages the shared memory arguments
//! (synchronizations and memory creations).
//!
//! Examples of libraries would be [BLAS][blas] or a special purpose Neural Network library like
//! [cuDNN][cudnn]. But unlike BLAS or cuDNN, a Collenchyma library is completely backend-agnostic
//! and can operate on any framework-supported hardware.
//!
//! Collenchyma ships with the most basic operations, but you should be able to easily write your
//! own backend-agnostic operations, too.
//!
//! [program]: ../program/index.html
//! [blas]: http://www.netlib.org/blas/
//! [cudnn]: https://developer.nvidia.com/cudnn

pub use self::numeric_helpers::Float;

#[macro_use]
pub mod blas;
/// Describes the Library numeric types and traits.
pub mod numeric_helpers {
    pub use num::traits::*;
}

#[derive(Debug, Copy, Clone)]
/// Defines a high-level library Error.
pub enum Error {
    /// Failure at a Blas Operation.
    Blas(::libraries::blas::Error),
}

impl ::std::fmt::Display for Error {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Error::Blas(ref err) => write!(f, "Blas error: {}", err),
        }
    }
}

impl ::std::error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Blas(ref err) => err.description(),
        }
    }

    fn cause(&self) -> Option<&::std::error::Error> {
        match *self {
            Error::Blas(ref err) => Some(err),
        }
    }
}

impl From<Error> for ::error::Error {
    fn from(err: Error) -> ::error::Error {
        ::error::Error::Operation(err)
    }
}
