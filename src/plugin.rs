//! Provides helpers for explicit implementations of Backend [Operations][operation].
//! [operation]: ../operation/index.html
//!
//! A Backend is a Rust struct like any other, therefore you probably would like to implement
//! certain methods for the Backend. As the whole purpose of a Backend is to provide an
//! abstraction over various computation devices and computation languages, these implemented
//! methods will than be able to excute on different devices and use the full power of the machine's
//! underlying hardware.
//!
//! So extending the Backend with operations is easy. In Collenchyma we call crates, which provide
//! operations for the Backend, Plugins. Plugins are usually a group of related operations of a common
//! field. Two examples for Collenchyma Plugins are [BLAS][collenchyma-blas] and [NN][ollenchyma-nn].
//!
//! A Plugin does roughly two important things. It provides generic traits and the explicit implementation
//! of these traits for one or (even better) all available Collenchyma Frameworks - common host CPU, OpenCL,
//! CUDA.
//!
//! The structure of Plugin is pretty easy with as less overhead as possible. Macros make implementations
//! even easier. If you would like to use specific Plugins for you Backend, all you need to do is,
//! setting them as dependencies in your Cargo file in addition to the Collenchyma crate. The Plugin
//! then automatically extends the Backend provided by Collenchyma.
//!
//! If you would like to extend the Backend with your own Plugin, than this is a straight forward process.
//! For now we recommend that you take a look at the general code structure of [Collenchyma-BLAS][collenchyma-blas]
//! or its documentation. Let us now about your Plugin on the Gitter chat, we are happy to feature
//! your Collenchyma Plugin on the README.
//!
//! [program]: ../program/index.html
//! [collenchyma-blas]: https://github.com/autumnai/collenchyma-blas
//! [collenchyma-nn]: https://github.com/autumnai/collenchyma-nn

pub use self::numeric_helpers::Float;

/// Describes numeric types and traits for a Plugin.
pub mod numeric_helpers {
    pub use num::traits::*;
}

#[derive(Debug, Copy, Clone)]
/// Defines a high-level Plugin Error.
pub enum Error {
    /// Failure at receiving the correct device memory from the SharedMemory.
    MissingMemoryForDevice(&'static str),
    /// Failure at the execution of the Operation.
    Operation(&'static str)
}

impl ::std::fmt::Display for Error {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Error::MissingMemoryForDevice(ref err) => write!(f, "MissingMemoryForDevice error: {}", err),
            Error::Operation(ref err) => write!(f, "Operation error: {}", err),
        }
    }
}

impl ::std::error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::MissingMemoryForDevice(ref err) => err,
            Error::Operation(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&::std::error::Error> {
        match *self {
            Error::MissingMemoryForDevice(_) => None,
            Error::Operation(_) => None,
        }
    }
}

impl From<Error> for ::error::Error {
    fn from(err: Error) -> ::error::Error {
        ::error::Error::Plugin(err)
    }
}
