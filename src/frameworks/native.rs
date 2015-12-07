//! Provides BLAS for a Native backend.

use ::operation::*;
use ::binary::*;
use ::library::*;
use collenchyma::device::DeviceType;
use collenchyma::backend::Backend;
use collenchyma::memory::MemoryType;
use collenchyma::frameworks::native::{Native, Function, Binary};
use collenchyma::plugin::Error;
use blas::{Asum, Axpy, Copy, Dot, Nrm2, Scal, Swap};

macro_rules! impl_binary(($($t: ident), +) => (
    $(
        impl IBlasBinary<$t> for Binary {
            type Asum = Function;
            type Axpy = Function;
            type Copy = Function;
            type Dot = Function;
            type Nrm2 = Function;
            type Scale = Function;
            type Swap = Function;

            fn asum(&self) -> Self::Asum {
                self.blas_asum
            }

            fn axpy(&self) -> Self::Axpy {
                self.blas_axpy
            }

            fn copy(&self) -> Self::Copy {
                self.blas_copy
            }

            fn dot(&self) -> Self::Dot {
                self.blas_dot
            }

            fn nrm2(&self) -> Self::Nrm2 {
                self.blas_nrm2
            }

            fn scale(&self) -> Self::Scale {
                self.blas_scale
            }

            fn swap(&self) -> Self::Swap {
                self.blas_swap
            }
        }
    )+
));

macro_rules! impl_asum_for {
    ($t:ident, $b:ty) => (
        impl IOperationAsum<$t> for $b {
            fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let mut r_slice = try!(result.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `result`."))).as_mut_slice::<$t>();
                r_slice[0] = Asum::asum(x_slice);
                Ok(())
            }
        }
    );
}

macro_rules! impl_axpy_for {
    ($t:ident, $b:ty) => (
        impl IOperationAxpy<$t> for $b {
            fn compute(&self, a: &MemoryType, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
                let a_slice = try!(a.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `a`."))).as_slice::<$t>();
                let x_slice = try!(x.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let y_slice = try!(y.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `y`."))).as_mut_slice::<$t>();
                Axpy::axpy(&a_slice[0], x_slice, y_slice);
                Ok(())
            }
        }
    );
}

macro_rules! impl_copy_for {
    ($t:ident, $b:ty) => (
        impl IOperationCopy<$t> for $b {
            fn compute(&self, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let y_slice = try!(y.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `y`."))).as_mut_slice::<$t>();
                Copy::copy(x_slice, y_slice);
                Ok(())
            }
        }
    );
}

macro_rules! impl_dot_for {
    ($t:ident, $b:ty) => (
        impl IOperationDot<$t> for $b {
            fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let y_slice = try!(y.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `y`."))).as_slice::<$t>();
                let mut r_slice = try!(result.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `result`."))).as_mut_slice::<$t>();
                r_slice[0] = Dot::dot(x_slice, y_slice);
                Ok(())
            }
        }
    );
}

macro_rules! impl_nrm2_for {
    ($t:ident, $b:ty) => (
        impl IOperationNrm2<$t> for $b {
            fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let mut r_slice = try!(result.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `result`."))).as_mut_slice::<$t>();
                r_slice[0] = Nrm2::nrm2(x_slice);
                Ok(())
            }
        }
    );
}

macro_rules! impl_scale_for {
    ($t:ident, $b:ty) => (
        impl IOperationScale<$t> for $b {
            fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error> {
                let a_slice = try!(a.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `a`."))).as_slice::<$t>();
                let mut x_slice = try!(x.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `x`."))).as_mut_slice::<$t>();
                Scal::scal(&a_slice[0], x_slice);
                Ok(())
            }
        }
    );
}

macro_rules! impl_swap_for {
    ($t:ident, $b:ty) => (
        impl IOperationSwap<$t> for $b {
            fn compute(&self, x: &mut MemoryType, y: &mut MemoryType) -> Result<(), Error> {
                let mut x_slice = try!(x.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `x`."))).as_mut_slice::<$t>();
                let mut y_slice = try!(y.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `y`."))).as_mut_slice::<$t>();
                Swap::swap(x_slice, y_slice);
                Ok(())
            }
        }
    );
}

macro_rules! impl_iblas_for {
    ($t:ident, $b:ty) => (
        impl_asum_for!($t, $b);
        impl_axpy_for!($t, $b);
        impl_copy_for!($t, $b);
        impl_dot_for!($t, $b);
        impl_nrm2_for!($t, $b);
        impl_scale_for!($t, $b);
        impl_swap_for!($t, $b);
    );
}

impl_binary!(f32, f64);
impl_iblas_for!(f32, Function);
impl_iblas_for!(f64, Function);

impl_iblas_for!(f32, Backend<Native>);
impl_iblas_for!(f64, Backend<Native>);

impl IBlas<f32> for Backend<Native> {
    type B = Binary;

    iblas_asum_for!(f32, Backend<Native>);
    iblas_axpy_for!(f32, Backend<Native>);
    iblas_copy_for!(f32, Backend<Native>);
    iblas_dot_for!(f32, Backend<Native>);
    iblas_nrm2_for!(f32, Backend<Native>);
    iblas_scale_for!(f32, Backend<Native>);
    iblas_swap_for!(f32, Backend<Native>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}

impl IBlas<f64> for Backend<Native> {
    type B = Binary;

    iblas_asum_for!(f64, Backend<Native>);
    iblas_axpy_for!(f64, Backend<Native>);
    iblas_copy_for!(f64, Backend<Native>);
    iblas_dot_for!(f64, Backend<Native>);
    iblas_nrm2_for!(f64, Backend<Native>);
    iblas_scale_for!(f64, Backend<Native>);
    iblas_swap_for!(f64, Backend<Native>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}
