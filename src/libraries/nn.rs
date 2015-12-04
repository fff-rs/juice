//! Provides backend-agnostic operations for Neural Networks.

use memory::MemoryType;
use shared_memory::SharedMemory;
use binary::IBinary;
use device::DeviceType;
use libraries::Float;

/// Provides the functionality for a backend to support Neural Network operations.
pub trait INN<F: Float> {
    /// The Binary representation for this Library.
    type B: INNBinary<F> + IBinary;

    /// Computes the absolute sum of vector `x`.
    ///
    /// Saves the result to `result`.
    /// This is a Level 1 BLAS operation.
    fn sigmoid(&self, x: &mut SharedMemory<F>, result: &mut SharedMemory<F>) -> Result<(), ::error::Error> {
        match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
        match result.add_device(self.device()) { _ => () }
        Ok(try!(
            self.binary().asum().compute(
                try!(x.get(self.device()).ok_or(Error::MissingArgument("Unable to resolve memory for `x`"))),
                try!(result.get_mut(self.device()).ok_or(Error::MissingArgument("Unable to resolve memory for `result`"))),
            )
        ))
    }

    /// Returns the binary representation
    fn binary(&self) -> &Self::B;

    /// Returns the device representation
    fn device(&self) -> &DeviceType;
}

/// Describes the operation binding for a NN Binary implementation.
pub trait INNBinary<F: Float> {
    /// Describes the Asum Operation.
    type Sigmoid: IOperationSigmoid<F>;

    /// Returns an initialized Asum operation.
    fn sigmoid(&self) -> Self::Sigmoid;
}

/// Describes a Sigmoid Operation.
pub trait IOperationSigmoid<F: Float> {
    /// Computes the Sigmoid operation.
    fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error>;
}

#[derive(Debug, Copy, Clone)]
/// Defines Blas Errors.
pub enum Error {
    /// Failure related to a missing argument.
    MissingArgument(&'static str),
    /// Failure related to an invalid argument.
    InvalidArgument(&'static str),
}

impl ::std::fmt::Display for Error {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Error::MissingArgument(ref err) => write!(f, "{:?}", err),
            Error::InvalidArgument(ref err) => write!(f, "{:?}", err),
        }
    }
}

impl ::std::error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::MissingArgument(ref err) => err,
            Error::InvalidArgument(ref err) => err,
        }
    }

    fn cause(&self) -> Option<&::std::error::Error> {
        match *self {
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
