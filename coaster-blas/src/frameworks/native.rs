//! Provides BLAS for a Native backend.

use crate::plugin::*;
use crate::transpose::*;
use coaster::backend::Backend;
use coaster::frameworks::native::Native;
use coaster::tensor::{ITensorDesc, SharedTensor};
use rblas;
use rblas::math::bandmat::BandMat;
use rblas::math::mat::Mat;
use rblas::matrix::Matrix;

macro_rules! read {
    ($x:ident, $t:ident, $slf:ident) => {
        $x.read($slf.device())?.as_slice::<$t>();
    };
}

macro_rules! read_write {
    ($x:ident, $t: ident, $slf:ident) => {
        $x.read_write($slf.device())?.as_mut_slice::<$t>();
    };
}

macro_rules! write_only {
    ($x:ident, $t: ident, $slf:ident) => {
        $x.write_only($slf.device())?.as_mut_slice::<$t>();
    };
}

macro_rules! iblas_asum_for_native {
    ($t:ident) => {
        fn asum(
            &self,
            x: &SharedTensor<$t>,
            result: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let r_slice = write_only!(result, $t, self);
            r_slice[0] = rblas::Asum::asum(read!(x, $t, self));
            Ok(())
        }
    };
}

macro_rules! iblas_axpy_for_native {
    ($t:ident) => {
        fn axpy(
            &self,
            a: &SharedTensor<$t>,
            x: &SharedTensor<$t>,
            y: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            rblas::Axpy::axpy(
                &read!(a, $t, self)[0],
                read!(x, $t, self),
                read_write!(y, $t, self),
            );
            Ok(())
        }
    };
}

macro_rules! iblas_copy_for_native {
    ($t:ident) => {
        fn copy(
            &self,
            x: &SharedTensor<$t>,
            y: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            rblas::Copy::copy(read!(x, $t, self), write_only!(y, $t, self));
            Ok(())
        }
    };
}

macro_rules! iblas_dot_for_native {
    ($t:ident) => {
        fn dot(
            &self,
            x: &SharedTensor<$t>,
            y: &SharedTensor<$t>,
            result: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let r_slice = write_only!(result, $t, self);
            r_slice[0] = rblas::Dot::dot(read!(x, $t, self), read!(y, $t, self));
            Ok(())
        }
    };
}

macro_rules! iblas_nrm2_for_native {
    ($t:ident) => {
        fn nrm2(
            &self,
            x: &SharedTensor<$t>,
            result: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let r_slice = write_only!(result, $t, self);
            r_slice[0] = rblas::Nrm2::nrm2(read!(x, $t, self));
            Ok(())
        }
    };
}

macro_rules! iblas_scal_for_native {
    ($t:ident) => {
        fn scal(
            &self,
            a: &SharedTensor<$t>,
            x: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            rblas::Scal::scal(&read!(a, $t, self)[0], read_write!(x, $t, self));
            Ok(())
        }
    };
}

macro_rules! iblas_swap_for_native {
    ($t:ident) => {
        fn swap(
            &self,
            x: &mut SharedTensor<$t>,
            y: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            rblas::Swap::swap(read_write!(x, $t, self), read_write!(y, $t, self));
            Ok(())
        }
    };
}

macro_rules! iblas_gbmv_for_native {
    ($t: ident) => {
        fn gbmv(
            &self,
            alpha: &SharedTensor<$t>,
            at: Transpose,
            a: &SharedTensor<$t>,
            kl: &SharedTensor<u32>,
            ku: &SharedTensor<u32>,
            x: &SharedTensor<$t>,
            beta: &SharedTensor<$t>,
            c: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let a_slice = read!(a, $t, self);
            let x_slice = read!(x, $t, self);
            let c_slice = read_write!(c, $t, self);

            // These values will always be u32
            let kl: u32 = read!(kl, u32, self)[0];
            let ku: u32 = read!(ku, u32, self)[0];

            let a_matrix = as_matrix(a_slice, a.desc().dims());
            let a_matrix = BandMat::from_matrix(a_matrix, kl, ku);

            rblas::Gbmv::gbmv(
                at.to_rblas(),
                &read!(alpha, $t, self)[0],
                &a_matrix,
                x_slice,
                &read!(beta, $t, self)[0],
                c_slice,
            );

            Ok(())
        }
    };
}

macro_rules! iblas_gemm_for_native {
    ($t:ident) => {
        fn gemm(
            &self,
            alpha: &SharedTensor<$t>,
            at: Transpose,
            a: &SharedTensor<$t>,
            bt: Transpose,
            b: &SharedTensor<$t>,
            beta: &SharedTensor<$t>,
            c: &mut SharedTensor<$t>,
        ) -> Result<(), ::coaster::error::Error> {
            let c_dims = c.desc().clone(); // FIXME: clone() can be removed

            let a_slice = read!(a, $t, self);
            let b_slice = read!(b, $t, self);
            let c_slice = write_only!(c, $t, self);

            let a_matrix = as_matrix(a_slice, a.desc().dims());
            let b_matrix = as_matrix(b_slice, b.desc().dims());
            let mut c_matrix = as_matrix(c_slice, &c_dims);
            rblas::Gemm::gemm(
                &read!(alpha, $t, self)[0],
                at.to_rblas(),
                &a_matrix,
                bt.to_rblas(),
                &b_matrix,
                &read!(beta, $t, self)[0],
                &mut c_matrix,
            );
            read_from_matrix(&c_matrix, c_slice);
            Ok(())
        }

        fn gemm_batched(
            &self,
            alpha: &SharedTensor<$t>,
            at: Transpose,
            a: &SharedTensor<$t>,
            bt: Transpose,
            b: &SharedTensor<$t>,
            beta: &SharedTensor<$t>,
            c: &mut SharedTensor<$t>,
            batch_count: usize
        ) -> Result<(), ::coaster::error::Error> {
        unimplemented!()
    }
    };
}

macro_rules! impl_iblas_for {
    ($t:ident, $b:ty) => {
        impl IBlas<$t> for $b {}

        // Level 1

        impl Asum<$t> for $b {
            iblas_asum_for_native!($t);
        }

        impl Axpy<$t> for $b {
            iblas_axpy_for_native!($t);
        }

        impl Copy<$t> for $b {
            iblas_copy_for_native!($t);
        }

        impl Dot<$t> for $b {
            iblas_dot_for_native!($t);
        }

        impl Nrm2<$t> for $b {
            iblas_nrm2_for_native!($t);
        }

        impl Scal<$t> for $b {
            iblas_scal_for_native!($t);
        }

        impl Swap<$t> for $b {
            iblas_swap_for_native!($t);
        }

        // Level 2

        impl Gbmv<$t> for $b {
            iblas_gbmv_for_native!($t);
        }

        // Level 3

        impl Gemm<$t> for $b {
            iblas_gemm_for_native!($t);
        }
    };
}

impl_iblas_for!(f32, Backend<Native>);
impl_iblas_for!(f64, Backend<Native>);

/// Create a rblas-Matrix from a slice and dimensions.
fn as_matrix<T: Clone + ::std::fmt::Debug>(slice: &[T], dims: &[usize]) -> Mat<T> {
    let n = dims[0];
    let m = dims.iter().skip(1).product();
    let mut mat: Mat<T> = Mat::new(n, m);
    for i in 0..n {
        for j in 0..m {
            let index = m * i + j;
            unsafe {
                *mat.as_mut_ptr().add(index) = slice[index].clone();
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
    use super::as_matrix;
    use coaster::backend::{Backend, IBackend};
    use coaster::frameworks::native::flatbox::FlatBox;
    use coaster::frameworks::Native;
    use coaster::tensor::SharedTensor;

    fn get_native_backend() -> Backend<Native> {
        Backend::<Native>::default().unwrap()
    }

    pub fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
        let mem_buffer = mem.as_mut_slice::<T>();
        for (index, datum) in data.iter().enumerate() {
            mem_buffer[index] = *datum;
        }
    }

    /// UTIL: as_matrix and read_from_matrix
    #[test]
    fn it_converts_correctly_to_and_from_matrix() {
        let backend = get_native_backend();
        let mut a = SharedTensor::<f32>::new(&vec![3, 2]);
        write_to_memory(
            a.write_only(backend.device()).unwrap(),
            &[2f32, 5f32, 2f32, 5f32, 2f32, 5f32],
        );

        {
            let a_slice_in = a.read(backend.device()).unwrap().as_slice::<f32>();
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
