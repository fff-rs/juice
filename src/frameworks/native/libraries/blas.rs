//! Provides BLAS for a Native backend.

use frameworks::native::{Function, Binary};
use memory::MemoryType;
use libraries::blas::*;
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

macro_rules! impl_asum(($($t: ident), +) => (
    $(
        impl IOperationAsum<$t> for Function {
            fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let mut r_slice = try!(result.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `result`."))).as_mut_slice::<$t>();
                r_slice[0] = Asum::asum(x_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_axpy(($($t: ident), +) => (
    $(
        impl IOperationAxpy<$t> for Function {
            fn compute(&self, a: &MemoryType, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
                let a_slice = try!(a.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `a`."))).as_slice::<$t>();
                let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let y_slice = try!(y.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `y`."))).as_mut_slice::<$t>();
                Axpy::axpy(&a_slice[0], x_slice, y_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_copy(($($t: ident), +) => (
    $(
        impl IOperationCopy<$t> for Function {
            fn compute(&self, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let y_slice = try!(y.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `y`."))).as_mut_slice::<$t>();
                Copy::copy(x_slice, y_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_dot(($($t: ident), +) => (
    $(
        impl IOperationDot<$t> for Function {
            fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let y_slice = try!(y.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `y`."))).as_slice::<$t>();
                let mut r_slice = try!(result.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `result`."))).as_mut_slice::<$t>();
                r_slice[0] = Dot::dot(x_slice, y_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_nrm2(($($t: ident), +) => (
    $(
        impl IOperationNrm2<$t> for Function {
            fn compute(&self, x: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `x`."))).as_slice::<$t>();
                let mut r_slice = try!(result.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `result`."))).as_mut_slice::<$t>();
                r_slice[0] = Nrm2::nrm2(x_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_scale(($($t: ident), +) => (
    $(
        impl IOperationScale<$t> for Function {
            fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error> {
                let a_slice = try!(a.as_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `a`."))).as_slice::<$t>();
                let mut x_slice = try!(x.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `x`."))).as_mut_slice::<$t>();
                Scal::scal(&a_slice[0], x_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_swap(($($t: ident), +) => (
    $(
        impl IOperationSwap<$t> for Function {
            fn compute(&self, x: &mut MemoryType, y: &mut MemoryType) -> Result<(), Error> {
                let mut x_slice = try!(x.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `x`."))).as_mut_slice::<$t>();
                let mut y_slice = try!(y.as_mut_native().ok_or(Error::InvalidArgument("Unable to receive native memory for `y`."))).as_mut_slice::<$t>();
                Swap::swap(x_slice, y_slice);
                Ok(())
            }
        }
    )+
));

impl_binary!(f32, f64);
impl_asum!(f32, f64);
impl_axpy!(f32, f64);
impl_copy!(f32, f64);
impl_dot!(f32, f64);
impl_nrm2!(f32, f64);
impl_scale!(f32, f64);
impl_swap!(f32, f64);
