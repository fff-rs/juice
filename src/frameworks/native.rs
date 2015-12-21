//! Provides BLAS for a Native backend.

use ::operation::*;
use ::plugin::*;
use ::transpose::*;
use collenchyma::backend::Backend;
use collenchyma::memory::MemoryType;
use collenchyma::frameworks::native::Native;
use collenchyma::plugin::Error;
use rblas::math::mat::Mat;
use rblas::matrix::Matrix;
use rblas::{Asum, Axpy, Copy, Dot, Nrm2, Scal, Swap}; // Level 1
use rblas::Gemm; // Level 3

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

macro_rules! impl_gemm_for {
    ($t:ident, $b:ty) => (
        impl IOperationGemm<$t> for $b {
            fn compute(&self, alpha: &MemoryType, at: Transpose, a_dims: &[usize], a: &MemoryType, bt: Transpose, b_dims: &[usize], b: &MemoryType, beta: &MemoryType, c_dims: &[usize], c: &mut MemoryType) -> Result<(), ::collenchyma::error::Error> {
                let alpha_slice = try!(alpha.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `alpha`."))).as_slice::<$t>();
                let a_slice = try!(a.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `a`."))).as_slice::<$t>();
                let beta_slice = try!(beta.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `beta`."))).as_slice::<$t>();
                let b_slice = try!(b.as_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `b`."))).as_slice::<$t>();
                let mut c_slice = try!(c.as_mut_native().ok_or(Error::MissingMemoryForDevice("Unable to receive native memory for `c`."))).as_mut_slice::<$t>();

                let a_matrix = as_matrix(a_slice, a_dims);
                let b_matrix = as_matrix(b_slice, b_dims);
                let mut c_matrix = as_matrix(c_slice, c_dims);
                Gemm::gemm(&alpha_slice[0], at.to_rblas(), &a_matrix, bt.to_rblas(), &b_matrix, &beta_slice[0], &mut c_matrix);
                read_from_matrix(&c_matrix, c_slice);
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

        impl_gemm_for!($t, $b);

        impl IBlas<$t> for $b {
            iblas_asum_for!($t, $b);
            iblas_axpy_for!($t, $b);
            iblas_copy_for!($t, $b);
            iblas_dot_for!($t, $b);
            iblas_nrm2_for!($t, $b);
            iblas_scale_for!($t, $b);
            iblas_swap_for!($t, $b);

            iblas_gemm_for!($t, $b);
        }
    );
}

impl_iblas_for!(f32, Backend<Native>);
impl_iblas_for!(f64, Backend<Native>);

/// Create a rblas-Matrix from a slice and dimensions.
fn as_matrix<T: Clone + ::std::fmt::Debug>(slice: &[T], dims: &[usize]) -> Mat<T> {
    let n = dims[0];
    let m = dims[1];
    let mut mat = Mat::new(n, m);
    for i in 0..n {
        for j in 0..m {
            unsafe {
                *mat.as_mut_ptr().offset((n*i + j) as isize) = slice[n*i + j].clone();
            }
        }
    }

    mat
}

fn read_from_matrix<T: Clone>(mat: &Mat<T>, slice: &mut [T]) {
    let n = mat.rows();
    let m = mat.cols();
    for i in 0..n {
        for j in 0..m {
            slice[n*i + j] = mat[i][j].clone();
        }
    }
}
