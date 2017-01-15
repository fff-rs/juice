#![cfg(feature = "native")] // required for data i/o

extern crate collenchyma_blas as co_blas;
extern crate collenchyma as co;

use std::fmt;
use co::backend::{Backend, BackendConfig, IBackend};
use co::framework::{IFramework};
use co::plugin::numeric_helpers::{cast, Float, NumCast};
use co::tensor::SharedTensor;
use co_blas::plugin::*;
use co_blas::transpose::Transpose;

#[cfg(feature = "cuda")]
use co::frameworks::Cuda;

pub fn get_native_backend() -> Backend<::co::frameworks::Native> {
    let framework = ::co::frameworks::Native::new();
    let hardwares = framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, &hardwares);
    Backend::new(backend_config).unwrap()
}

#[cfg(feature = "cuda")]
pub fn get_cuda_backend() -> Backend<Cuda> {
    let framework = Cuda::new();
    let hardwares = framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, &hardwares);
    Backend::new(backend_config).unwrap()
}


pub fn write_to_tensor<T, S>(x: &mut SharedTensor<T>, data: &[S])
    where T: ::std::marker::Copy + NumCast,
          S: ::std::marker::Copy + NumCast {

    let native = get_native_backend();
    let mem = x.write_only(native.device()).unwrap().as_mut_native().unwrap();
    let mut mem_buffer = mem.as_mut_slice::<T>();
    for (i, x) in data.iter().enumerate() {
        mem_buffer[i] = cast::<S, T>(*x).unwrap();
    }
}

pub fn tensor_assert_eq<T: ::std::fmt::Debug>(x: &SharedTensor<T>, data: &[f64])
    where T: fmt::Debug + PartialEq + NumCast {
    let native = get_native_backend();
    let mem = x.read(native.device()).unwrap().as_native().unwrap();
    let mem_slice = mem.as_slice::<T>();

    assert_eq!(mem_slice.len(), data.len());
    for (x1, x2) in mem_slice.iter().zip(data.iter()) {
        let x2_t = cast::<f64, T>(*x2).unwrap();
        assert_eq!(*x1, x2_t); // TODO: compare approximately
    }
}

pub fn test_asum<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Asum<T> + IBackend {
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut result = SharedTensor::<T>::new(&[1]);

    write_to_tensor(&mut x, &[1, -2, 3]);
    backend.asum(&x, &mut result).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&result, &[6.0]);
}

pub fn test_axpy<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Axpy<T> + IBackend {
    let mut a = SharedTensor::<T>::new(&[1]);
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut a, &[2]);
    write_to_tensor(&mut x, &[1, 2, 3]);
    write_to_tensor(&mut y, &[1, 2, 3]);

    backend.axpy(&a, &x, &mut y).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&y, &[3.0, 6.0, 9.0]);
}

pub fn test_copy<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Copy<T> + IBackend {
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut x, &[1, 2, 3]);

    backend.copy(&x, &mut y).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&y, &[1.0, 2.0, 3.0]);
}

pub fn test_dot<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Dot<T> + IBackend {
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    let mut result = SharedTensor::<T>::new(&[1]);
    write_to_tensor(&mut x, &[1, 2, 3]);
    write_to_tensor(&mut y, &[1, 2, 3]);

    backend.dot(&x, &y, &mut result).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&result, &[14.0]);
}

pub fn test_nrm2<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Nrm2<T> + IBackend {
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut result = SharedTensor::<T>::new(&[1]);
    write_to_tensor(&mut x, &[1, 2, 2]);

    backend.nrm2(&x, &mut result).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&result, &[3.0]);
}

pub fn test_scal<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Scal<T> + IBackend {
    let mut a = SharedTensor::<T>::new(&[1]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut a, &[2]);
    write_to_tensor(&mut y, &[1, 2, 3]);

    backend.scal(&a, &mut y).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&y, &[2.0, 4.0, 6.0]);
}

pub fn test_swap<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Swap<T> + IBackend {
    let mut x = SharedTensor::<T>::new(&[3]);
    let mut y = SharedTensor::<T>::new(&[3]);
    write_to_tensor(&mut x, &[1, 2, 3]);
    write_to_tensor(&mut y, &[3, 2, 1]);

    backend.swap(&mut x, &mut y).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&x, &[3.0, 2.0, 1.0]);
    tensor_assert_eq(&y, &[1.0, 2.0, 3.0]);
}

pub fn test_gemm<T, F>(backend: Backend<F>)
    where T: Float + fmt::Debug,
            F: IFramework,
            Backend<F>: Gemm<T> + IBackend {
    let mut alpha = SharedTensor::<T>::new(&[1]);
    let mut beta = SharedTensor::<T>::new(&[1]);
    let mut a = SharedTensor::<T>::new(&[3, 2]);
    let mut b = SharedTensor::<T>::new(&[2, 3]);
    write_to_tensor(&mut alpha, &[1]);
    write_to_tensor(&mut beta, &[0]);
    write_to_tensor(&mut a, &[2, 5, 2,  5, 2, 5]);
    write_to_tensor(&mut b, &[4, 1, 1,  4, 1, 1]);

    let mut c = SharedTensor::<T>::new(&[3, 3]);
    backend.gemm(&alpha,
                    Transpose::NoTrans, &a,
                    Transpose::NoTrans, &b,
                    &beta,
                    &mut c).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&c, &[
        28.0, 7.0, 7.0,
        28.0, 7.0, 7.0,
        28.0, 7.0, 7.0]);

    let mut d = SharedTensor::<T>::new(&[2, 2]);
    backend.gemm(&alpha,
                    Transpose::Trans, &a,
                    Transpose::Trans, &b,
                    &beta,
                    &mut d).unwrap();
    backend.synchronize().unwrap();
    tensor_assert_eq(&d, &[12.0, 12.0, 30.0, 30.0]);
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

test_blas!(native_f32, get_native_backend, f32);
test_blas!(native_f64, get_native_backend, f64);

#[cfg(feature = "cuda")]
test_blas!(cuda_f32, get_cuda_backend, f32);
