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

use memory::MemoryType;
use shared_memory::SharedMemory;
use binary::IBinary;
use device::DeviceType;

/// Provides the functionality for a backend to support Basic Linear Algebra Subprogram operations.
pub trait IBlas {
    /// The Binary representation for this Library.
    type B: IBlasBinary + IBinary;

    /// Level 1 operation
    fn dot(&self, x: &mut SharedMemory<f32>, y: &mut SharedMemory<f32>, result: &mut SharedMemory<f32>) -> Result<(), ::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().dot().compute::<f32>(
                try!(x.get(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `x`")))),
                try!(y.get(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `y`")))),
                try!(result.get_mut(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `result`")))),
            )
        ))
    }

    /// Returns the binary representation
    fn binary(&self) -> Self::B;

    /// Returns the device representation
    fn device(&self) -> &DeviceType;
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
    fn compute<T>(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}

#[derive(Debug)]
/// Defines Blas Errors.
pub enum Error {
    /// Failure related to a Dot operation.
    Dot(String),
    /// Failure related to a missing argument.
    MissingArgument(String),
    /// Failure related to an invalid argument.
    InvalidArgument(String),
}

impl ::std::fmt::Display for Error {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Error::Dot(ref err) => write!(f, "{:?}", err),
            Error::MissingArgument(ref err) => write!(f, "{:?}", err),
            Error::InvalidArgument(ref err) => write!(f, "{:?}", err),
        }
    }
}

impl ::std::error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::Dot(ref err) => err,
            Error::MissingArgument(ref err) => err,
            Error::InvalidArgument(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&::std::error::Error> {
        match *self {
            Error::Dot(_) => None,
            Error::MissingArgument(_) => None,
            Error::InvalidArgument(_) => None,
        }
    }
}

impl From<Error> for ::libraries::Error {
    fn from(err: Error) -> ::libraries::Error {
        ::libraries::Error::Blas(err)
    }
}

impl From<Error> for ::error::Error {
    fn from(err: Error) -> ::error::Error {
        ::error::Error::Operation(From::from(err))
    }
}
