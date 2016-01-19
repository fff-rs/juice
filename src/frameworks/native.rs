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
    let m = dims.iter().skip(1).fold(1, |prod, i| prod * i);
    let mut mat: Mat<T> = Mat::new(n, m);
    for i in 0..n {
        for j in 0..m {
            let index = m * i + j;
            unsafe {
                *mat.as_mut_ptr().offset(index as isize) = slice[index].clone();
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
            let index = m * i + j;
            slice[index] = mat[i][j].clone();
        }
    }
}

#[cfg(test)]
mod test {
    use collenchyma::backend::{Backend, BackendConfig};
    use collenchyma::framework::IFramework;
    use collenchyma::frameworks::Native;
    use collenchyma::tensor::SharedTensor;
    use collenchyma::memory::MemoryType;
    use super::as_matrix;

    fn get_native_backend() -> Backend<Native> {
        let framework = Native::new();
        let hardwares = framework.hardwares().to_vec();
        let backend_config = BackendConfig::new(framework, &hardwares);
        Backend::new(backend_config).unwrap()
    }

    pub fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
        match mem {
            &mut MemoryType::Native(ref mut mem) => {
                let mut mem_buffer = mem.as_mut_slice::<T>();
                for (index, datum) in data.iter().enumerate() {
                    mem_buffer[index] = *datum;
                }
            },
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => assert!(false)
        }
    }

    /// UTIL: as_matrix and read_from_matrix
    #[test]
    fn it_converts_correctly_to_and_from_matrix() {
        let backend = get_native_backend();
        let mut a = SharedTensor::<f32>::new(backend.device(), &vec![3, 2]).unwrap();
        write_to_memory(a.get_mut(backend.device()).unwrap(),
            &[2f32, 5f32,
              2f32, 5f32,
              2f32, 5f32]);

        // let mut b = SharedTensor::<f32>::new(backend.device(), &vec![2, 3]).unwrap();
        // write_to_memory(b.get_mut(backend.device()).unwrap(),
        //     &[4f32, 1f32, 1f32,
        //       4f32, 1f32, 1f32]);

        // let mut alpha = SharedTensor::<f32>::new(backend.device(), &vec![1]).unwrap();
        // write_to_memory(alpha.get_mut(backend.device()).unwrap(), &[1f32]);
        //
        // let mut beta = SharedTensor::<f32>::new(backend.device(), &vec![1]).unwrap();
        // write_to_memory(beta.get_mut(backend.device()).unwrap(), &[0f32]);
        {
            let a_slice_in = a.get(backend.device()).unwrap().as_native().unwrap().as_slice::<f32>();
            let a_mat = as_matrix(a_slice_in, &[3, 2]);
            // right
            assert_eq!(a_mat[0][0], 2f32);
            assert_eq!(a_mat[0][1], 5f32);
            assert_eq!(a_mat[1][0], 2f32);
            assert_eq!(a_mat[1][1], 5f32);
            assert_eq!(a_mat[2][0], 2f32);
            assert_eq!(a_mat[2][1], 5f32);
        }
    }
}
