#![cfg(feature = "native")] // required for data i/o

extern crate coaster as co;
extern crate coaster_blas as co_blas;

use crate::co::backend::{Backend, IBackend};
use crate::co::framework::IFramework;
use std::fmt;

use crate::co::plugin::numeric_helpers::{cast, Float, NumCast};
use crate::co::tensor::{ITensorDesc, SharedTensor};
use crate::co_blas::plugin::*;
use crate::co_blas::transpose::Transpose;

#[cfg(feature = "native")]
use crate::co::frameworks::Native;

#[cfg(feature = "cuda")]
use crate::co::frameworks::Cuda;

#[cfg(feature = "native")]
fn get_native_backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}
#[cfg(feature = "cuda")]
use crate::co::frameworks::cuda::get_cuda_backend;

// #[cfg(feature = "opencl")]
// fn get_opencl_backend() -> Backend<OpenCL> {
//     Backend::<OpenCL>::default().unwrap()
// }

// TODO reuse the coaster-nn methods
pub fn write_to_tensor<T>(xs: &mut SharedTensor<T>, data: &[f64])
where
    T: ::std::marker::Copy + NumCast,
{
    assert_eq!(xs.desc().size(), data.len());
    let native = get_native_backend();
    let native_dev = native.device();
    {
        let mem = xs.write_only(native_dev).unwrap();
        let mem_buffer = mem.as_mut_slice::<T>();
        for (i, x) in data.iter().enumerate() {
            mem_buffer[i] = cast::<_, T>(*x).unwrap();
        }
    }
}

// TODO reuse the coaster-nn methods
pub fn tensor_assert_eq<T>(xs: &SharedTensor<T>, data: &[f64], epsilon_mul: f64)
where
    T: Float + fmt::Debug + PartialEq + NumCast,
{
    let e = 0. * epsilon_mul;

    let native = get_native_backend();
    let native_dev = native.device();

    let mem = xs.read(native_dev).unwrap();
    let mem_slice = mem.as_slice::<T>();

    assert_eq!(mem_slice.len(), data.len());
    for (x1, x2) in mem_slice.iter().zip(data.iter()) {
        let x1_t = cast::<_, f64>(*x1).unwrap();
        let diff = (x1_t - x2).abs();
        let max_diff = e * (x1_t.abs() + x2.abs()) * 0.5;
        if (x1_t - x2).abs() > e * (x1_t.abs() + x2.abs()) * 0.5 {
            println!(
                "Results differ: {:?} != {:?} ({:.2?} in {:?} and {:?}",
                x1_t,
                x2,
                diff / max_diff,
                mem_slice,
                data
            );
            assert!(false);
        }
    }
}

pub fn test_asum<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Asum<T> + IBackend,
{
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut result = SharedTensor::<T>::new(&[1]);
    write_to_tensor(&mut x, &[1.0, -2.0, 3.0]);

    backend.asum(&x, &mut result).unwrap();
    tensor_assert_eq(&result, &[6.0], 0.);
}

pub fn test_axpy<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Axpy<T> + IBackend,
{
    let mut a = SharedTensor::<T>::new(&[1]);
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut a, &[2.]);
    write_to_tensor(&mut x, &[1., 2., 3.]);
    write_to_tensor(&mut y, &[1., 2., 3.]);

    backend.axpy(&a, &x, &mut y).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&y, &[3.0, 6.0, 9.0], 0.);
}

pub fn test_copy<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Copy<T> + IBackend,
{
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut x, &[1., 2., 3.]);

    backend.copy(&x, &mut y).unwrap();
    tensor_assert_eq(&y, &[1.0, 2.0, 3.0], 0.);
}

pub fn test_dot<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Dot<T> + IBackend,
{
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    let mut result = SharedTensor::<T>::new(&[1]);
    write_to_tensor(&mut x, &[1., 2., 3.]);
    write_to_tensor(&mut y, &[1., 2., 3.]);

    backend.dot(&x, &y, &mut result).unwrap();
    tensor_assert_eq(&result, &[14.0], 0.);
}

pub fn test_nrm2<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Nrm2<T> + IBackend,
{
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut result = SharedTensor::<T>::new(&[1]);
    write_to_tensor(&mut x, &[1., 2., 2.]);

    backend.nrm2(&x, &mut result).unwrap();
    tensor_assert_eq(&result, &[3.0], 0.);
}

pub fn test_scal<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Scal<T> + IBackend,
{
    let mut a = SharedTensor::<T>::new(&[1]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut a, &[2.]);
    write_to_tensor(&mut y, &[1., 2., 3.]);

    backend.scal(&a, &mut y).unwrap();

    tensor_assert_eq(&y, &[2.0, 4.0, 6.0], 0.);
}

pub fn test_swap<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Swap<T> + IBackend,
{
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut x, &[1., 2., 3.]);
    write_to_tensor(&mut y, &[3., 2., 1.]);

    backend.swap(&mut x, &mut y).unwrap();

    tensor_assert_eq(&x, &[3.0, 2.0, 1.0], 0.);
    tensor_assert_eq(&y, &[1.0, 2.0, 3.0], 0.);
}

pub fn test_gbmv1<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Gbmv<T> + IBackend,
{
    let mut alpha = SharedTensor::<T>::new(&[1]);
    let mut beta = SharedTensor::<T>::new(&[1]);
    let mut a = SharedTensor::<T>::new(&[4, 4]);
    let mut x = SharedTensor::<T>::new(&[4]);
    let mut y = SharedTensor::<T>::new(&[4]);

    let mut kl = SharedTensor::<u32>::new(&[1]);
    let mut ku = SharedTensor::<u32>::new(&[1]);

    write_to_tensor(&mut alpha, &[1.]);
    write_to_tensor(&mut beta, &[3.]);

    /*
     * The band matrix should look like this
    write_to_tensor(&mut a,
        &[
          0.0, 0.5, 2.0,
          1.0, 0.5, 2.0,
          1.0, 0.5, 2.0,
          1.0, 0.5, 0.0,
          0.0, 0.0, 0.0,
          0.0
        ]);
    */

    write_to_tensor(
        &mut a,
        &[
            0.5, 2.0, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 0.0, 1.0, 0.5,
        ],
    );

    write_to_tensor(&mut x, &[1., 2., 2., 1.]);
    write_to_tensor(&mut y, &[0.5, 1., 2., 3.]);

    write_to_tensor(&mut kl, &[1.0]);
    write_to_tensor(&mut ku, &[1.0]);

    backend
        .gbmv(&alpha, Transpose::NoTrans, &a, &kl, &ku, &x, &beta, &mut y)
        .unwrap();

    tensor_assert_eq(&y, &[6.0, 9.0, 11., 11.5], 0.5);
}

pub fn test_gbmv2<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Gbmv<T> + IBackend,
{
    let mut alpha = SharedTensor::<T>::new(&[1]);
    let mut beta = SharedTensor::<T>::new(&[1]);
    let mut a = SharedTensor::<T>::new(&[4, 7]);
    let mut x = SharedTensor::<T>::new(&[7]);
    let mut y = SharedTensor::<T>::new(&[4]);

    let mut kl = SharedTensor::<u32>::new(&[1]);
    let mut ku = SharedTensor::<u32>::new(&[1]);

    write_to_tensor(&mut alpha, &[1.]);
    write_to_tensor(&mut beta, &[0.]);

    write_to_tensor(
        &mut a,
        &[
            0.5, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5,
            2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 2.0, 3.0, 0.0,
        ],
    );

    write_to_tensor(&mut x, &[1., 2., 3., 4., 5., 6., 7.]);
    write_to_tensor(&mut y, &[0., 0., 0., 0.]);

    write_to_tensor(&mut kl, &[1.0]);
    write_to_tensor(&mut ku, &[2.0]);

    backend
        .gbmv(&alpha, Transpose::NoTrans, &a, &kl, &ku, &x, &beta, &mut y)
        .unwrap();

    tensor_assert_eq(&y, &[13.5, 20.0, 26.5, 33.0], 0.5);
}

pub fn test_gemm<T, F>(backend: Backend<F>)
where
    T: Float + fmt::Debug,
    F: IFramework,
    Backend<F>: Gemm<T> + IBackend,
{
    let mut alpha = SharedTensor::<T>::new(&[1]);
    let mut beta = SharedTensor::<T>::new(&[1]);
    let mut a = SharedTensor::<T>::new(&[3, 2]);
    let mut b = SharedTensor::<T>::new(&[2, 3]);
    write_to_tensor(&mut alpha, &[1.]);
    write_to_tensor(&mut beta, &[0.]);
    write_to_tensor(&mut a, &[2., 5., 2., 5., 2., 5.]);
    write_to_tensor(&mut b, &[4., 1., 1., 4., 1., 1.]);

    let mut c = SharedTensor::<T>::new(&[3, 3]);
    backend
        .gemm(
            &alpha,
            Transpose::NoTrans,
            &a,
            Transpose::NoTrans,
            &b,
            &beta,
            &mut c,
        )
        .unwrap();

    tensor_assert_eq(&c, &[28.0, 7.0, 7.0, 28.0, 7.0, 7.0, 28.0, 7.0, 7.0], 0.);

    let mut d = SharedTensor::<T>::new(&[2, 2]);
    backend
        .gemm(
            &alpha,
            Transpose::Trans,
            &a,
            Transpose::Trans,
            &b,
            &beta,
            &mut d,
        )
        .unwrap();

    tensor_assert_eq(&d, &[12.0, 12.0, 30.0, 30.0], 0.);
}

macro_rules! test_blas {
    ($mod_name:ident, $backend_getter:ident, $t:ident) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn it_computes_correct_asum() {
                test_asum::<$t, _>($backend_getter());
            }

            #[test]
            fn it_computes_correct_axpy() {
                test_axpy::<$t, _>($backend_getter());
            }

            #[test]
            fn it_computes_correct_copy() {
                test_copy::<$t, _>($backend_getter());
            }

            #[test]
            fn it_computes_correct_dot() {
                test_dot::<$t, _>($backend_getter());
            }

            #[test]
            fn it_computes_correct_nrm2() {
                test_nrm2::<$t, _>($backend_getter());
            }


            #[test]
            fn it_computes_correct_scal() {
                test_scal::<$t, _>($backend_getter());
            }

            #[test]
            fn it_computes_correct_swap() {
                test_swap::<$t, _>($backend_getter());
            }

            #[test]
            fn it_computes_correct_gemm() {
                test_gemm::<$t, _>($backend_getter());
            }
        }
    };
}

macro_rules! test_blas_gbmv {
    ($mod_name:ident, $backend_getter:ident, $t:ident) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn it_computes_correct_gbmv_square() {
                test_gbmv1::<$t, _>($backend_getter());
            }

            #[test]
            fn it_computes_correct_gbmv_nonsquare() {
                test_gbmv2::<$t, _>($backend_getter());
            }
        }
    };
}

test_blas!(native_f32, get_native_backend, f32);
test_blas!(native_f64, get_native_backend, f64);

// This is temporary until CUDA impelements Gbmv trait
test_blas_gbmv!(native_f32_gbmv, get_native_backend, f32);
test_blas_gbmv!(native_f64_gbmv, get_native_backend, f64);

#[cfg(feature = "cuda")]
test_blas!(cuda_f32, get_cuda_backend, f32);
