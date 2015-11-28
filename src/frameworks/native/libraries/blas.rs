//! Provides BLAS for a Native backend.

use frameworks::native::{Function, Binary};
use memory::MemoryType;
use libraries::blas::*;
use blas::{Dot, Scal, Axpy};

macro_rules! impl_binary(($($t: ident), +) => (
    $(
        impl IBlasBinary<$t> for Binary {
            type Dot = Function;
            type Scale = Function;
            type Axpy = Function;

            fn dot(&self) -> Self::Dot {
                self.blas_dot
            }

            fn scale(&self) -> Self::Scale {
                self.blas_scale
            }

            fn axpy(&self) -> Self::Scale {
                self.blas_axpy
            }
        }
    )+
));

macro_rules! impl_dot(($($t: ident), +) => (
    $(
        impl IOperationDot<$t> for Function {
            fn compute(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
                let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `x`.")))).as_slice::<$t>();
                let y_slice = try!(y.as_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `y`.")))).as_slice::<$t>();
                let mut r_slice = try!(result.as_mut_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `result`.")))).as_mut_slice::<$t>();
                r_slice[0] = Dot::dot(x_slice, y_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_scale(($($t: ident), +) => (
    $(
        impl IOperationScale<$t> for Function {
            fn compute(&self, a: &MemoryType, x: &mut MemoryType) -> Result<(), Error> {
                let a_slice = try!(a.as_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `a`.")))).as_slice::<$t>();
                let mut x_slice = try!(x.as_mut_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `x`.")))).as_mut_slice::<$t>();
                Scal::scal(&a_slice[0], x_slice);
                Ok(())
            }
        }
    )+
));

macro_rules! impl_axpy(($($t: ident), +) => (
    $(
        impl IOperationAxpy<$t> for Function {
            fn compute(&self, a: &MemoryType, x: &MemoryType, y: &mut MemoryType) -> Result<(), Error> {
                let a_slice = try!(a.as_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `a`.")))).as_slice::<$t>();
                let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `x`.")))).as_slice::<$t>();
                let y_slice = try!(y.as_mut_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `y`.")))).as_mut_slice::<$t>();
                Axpy::axpy(&a_slice[0], x_slice, y_slice);
                Ok(())
            }
        }
    )+
));

impl_binary!(f32, f64);
impl_dot!(f32, f64);
impl_scale!(f32, f64);
impl_axpy!(f32, f64);
