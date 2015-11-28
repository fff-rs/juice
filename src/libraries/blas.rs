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
use num::traits::Float;

/// Provides the functionality for a backend to support Basic Linear Algebra Subprogram operations.
pub trait IBlas<F: Float> {
    /// The Binary representation for this Library.
    type B: IBlasBinary<F> + IBinary;

    /// Computes the [dot product][dot-product] over x and y.
    /// [dot-product]: https://en.wikipedia.org/wiki/Dot_product
    ///
    /// Saves the resulting value into `result`.
    /// This is a Level 1 BLAS operation.
    fn dot(&self, x: &mut SharedMemory<F>, y: &mut SharedMemory<F>, result: &mut SharedMemory<F>) -> Result<(), ::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().dot().compute(
                try!(x.get(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `x`")))),
                try!(y.get(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `y`")))),
                try!(result.get_mut(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `result`")))),
            )
        ))
    }

    /// Scales a vector `x` by a constant `a` aka. `a * x`.
    ///
    /// Saves the resulting vector back into `x`.
    /// This is a Level 1 BLAS operation.
    fn scale(&self, a: &mut SharedMemory<F>, x: &mut SharedMemory<F>) -> Result<(), ::error::Error> {
        match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        Ok(try!(
            self.binary().scale().compute(
                try!(a.get(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `a`")))),
                try!(x.get_mut(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `x`")))),
            )
        ))
    }

    /// Computes a vector `x` times a constant `a` plus a vector `y` aka. `a * x + y`.
    ///
    /// Saves the resulting vector back into `y`.
    /// This is a Level 1 BLAS operation.
    fn axpy(&self, a: &mut SharedMemory<F>, x: &mut SharedMemory<F>, y: &mut SharedMemory<F>) -> Result<(), ::error::Error> {
        match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match y.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        Ok(try!(
            self.binary().axpy().compute(
                try!(a.get(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `a`")))),
                try!(x.get(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `x`")))),
                try!(y.get_mut(self.device()).ok_or(Error::MissingArgument(format!("Unable to resolve memory for `y`")))),
            )
        ))
    }

    /// Returns the binary representation
    fn binary(&self) -> Self::B;

    /// Returns the device representation
    fn device(&self) -> &DeviceType;
}

/// Describes the operation binding for a Blas Binary implementation.
pub trait IBlasBinary<F: Float> {
    /// Describes the Dot Operation.
    type Dot: IOperationDot<F>;
    /// Describes the Scale Operation.
    type Scale: IOperationScale<F>;
    /// Describes the Axpy Operation.
    type Axpy: IOperationAxpy<F>;

    /// Returns an initialized Dot operation.
    fn dot(&self) -> Self::Dot;
    /// Returns an initialized Scale operation.
    fn scale(&self) -> Self::Scale;
    /// Returns an initialized Axpy operation.
    fn axpy(&self) -> Self::Axpy;
}

/// Describes a Dot Operation.
pub trait IOperationDot<F: Float> {
    /// Computes the Dot operation.
    fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Scale Operation.
pub trait IOperationScale<F: Float> {
    /// Computes the Scale operation.
    fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error>;
}

/// Describes a Axpy Operation.
pub trait IOperationAxpy<F: Float> {
    /// Computes the Axpy operation.
    fn compute(&self, a: &MemoryType, x: & MemoryType, y: &mut MemoryType) -> Result<(), Error>;
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
