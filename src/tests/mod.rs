//! This module defines common functions used by tests.
//! Test vectors for ReLU, Sigmoid, Tanh, Softmax, LogSoftmax were generated
//! by [nn-test-vectors][nn-test-vectors] script. It uses arbitrary precision
//! floating point library, so if there is need to increase precision for f128,
//! it can be easily done.
//!
//! [nn-test-vectors]: https://github.com/autumnai/collenchyma

use std;
use std::fmt;

use rand::thread_rng;
use rand::distributions::{range, IndependentSample, Range};

use co::prelude::*;
use co::plugin::numeric_helpers::{cast, NumCast};

pub trait Epsilon {
    fn epsilon() -> Self;
}

impl Epsilon for f32 {
    fn epsilon() -> Self { std::f32::EPSILON }
}

impl Epsilon for f64 {
    fn epsilon() -> Self { std::f64::EPSILON }
}


#[cfg(feature = "native")]
fn get_native_backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}
#[cfg(feature = "cuda")]
fn get_cuda_backend() -> Backend<Cuda> {
    Backend::<Cuda>::default().unwrap()
}
#[cfg(feature = "opencl")]
fn get_opencl_backend() -> Backend<OpenCL> {
    Backend::<OpenCL>::default().unwrap()
}

pub fn write_to_tensor<T,F>(backend: &Backend<F>, xs: &mut SharedTensor<T>, data: &[f64])
    where T: ::std::marker::Copy + NumCast,
          F: IFramework,
          Backend<F>: IBackend {

    assert_eq!(xs.desc().size(), data.len());
    let native = get_native_backend();
    let native_dev = native.device();
    {
        let mem = xs.write_only(native_dev).unwrap().as_mut_native().unwrap();
        let mut mem_buffer = mem.as_mut_slice::<T>();
        for (i, x) in data.iter().enumerate() {
            mem_buffer[i] = cast::<_, T>(*x).unwrap();
        }
    }


    let other_dev = backend.device();
    match other_dev {
        &DeviceType::Native(_) => {}
        _ => {
                // sync now TODO probably pointless with autosync
                xs.read(&other_dev).unwrap();
                xs.drop_device(&native_dev).unwrap();
                }
    }
}

pub fn filled_tensor<T,F>(backend: &Backend<F>, dims: &[usize], data: &[f64]) -> SharedTensor<T>
    where T: ::std::marker::Copy + NumCast,
          F: IFramework,
          Backend<F>: IBackend {

    let mut x = SharedTensor::new(&dims);
    write_to_tensor(backend, &mut x, data);
    x
}

// Currently unused. It was supposed to be used for random tests with inlined
// verification or cross tests (Native <-> Cuda), but they aren't implemented
// yet.
pub fn uniformly_random_tensor<T,F>(backend: &Backend<F>, dims: &[usize], low: T, high: T) -> SharedTensor<T>
    where T: Copy + PartialEq + PartialOrd + range::SampleRange,
          F: IFramework,
          Backend<F>: IBackend {

    let mut xs = SharedTensor::new(&dims);
    {
        let native = get_native_backend();
        let native_dev = native.device();
        let other_dev = backend.device();
        {
            let mut mem = xs.write_only(native_dev).unwrap().as_mut_native().unwrap();
            let mem_slice = mem.as_mut_slice::<T>();

            let mut rng = thread_rng();
            let distr = Range::new(low, high);
            for x in mem_slice {
                *x = distr.ind_sample(&mut rng);
            }
        }
        match other_dev {
            &DeviceType::Native(_) => {}
            _ => {
                    // sync now TODO probably pointless with autosync
                    xs.read(&other_dev).unwrap();
                    xs.drop_device(&native_dev).unwrap();
                    }
        }
    }
    xs
}

/// This function tests that contents of `xs` and `data` don't differ too much.
/// They are allowed to differ by no more than `x * T::epsilon() * epsilon_mul`.
/// Of course if there were inevitable substantial rounding errors during
/// calculations of `xs` there may be false positives.
pub fn tensor_assert_eq<T,F>(backend: &Backend<F>, xs: &SharedTensor<T>, data: &[f64], epsilon_mul: f64)
    where T: Copy + fmt::Debug + PartialEq + NumCast + Epsilon,
          F: IFramework ,
          Backend<F>: IBackend {

    let e = cast::<_, f64>(T::epsilon()).unwrap() * epsilon_mul;

    let native = get_native_backend();
    let native_dev = native.device();
    let other_dev = backend.device();

    match other_dev {
        &DeviceType::Native(_) => {}
        _ => { xs.read(&native_dev).unwrap(); }
    }

    let mem = xs.read(&native_dev).unwrap().as_native().unwrap();
    let mem_slice = mem.as_slice::<T>();

    assert_eq!(mem_slice.len(), data.len());
    for (x1, x2) in mem_slice.iter().zip(data.iter()) {
        let x1_t = cast::<_, f64>(*x1).unwrap();
        let diff = (x1_t - x2).abs();
        let max_diff = e * (x1_t.abs() + x2.abs()) * 0.5;
        if (x1_t - x2).abs() > e * (x1_t.abs() + x2.abs()) * 0.5 {
            println!("Results differ: {:?} != {:?} ({:.2?} in {:?} and {:?}",
                     x1_t, x2, diff / max_diff, mem_slice, data);
            assert!(false);
        }
    }
}

// All operations for Cuda and Native are provided for f32 and f64.
// Those macros remove boilerplate in test definitions.
// concat_idents! is behind feature gate at the moment, otherwise
// invocations could be made much less verbose.
macro_rules! test_cuda {
    ($test_name:ident, $f32_name:ident, $f64_name:ident) => {

        #[cfg(feature = "cuda")]
        #[test]
        fn $f32_name() {
            $test_name::<f32, _>(::tests::get_cuda_backend())
        }

        #[cfg(feature = "cuda")]
        #[test]
        fn $f64_name() {
            $test_name::<f64, _>(::tests::get_cuda_backend())
        }
    }
}

macro_rules! test_native {
    ($test_name:ident, $f32_name:ident, $f64_name:ident) => {

        #[cfg(feature = "native")]
        #[test]
        fn $f32_name() {
            $test_name::<f32, _>(::tests::get_native_backend())
        }

        #[cfg(feature = "native")]
        #[test]
        fn $f64_name() {
            $test_name::<f64, _>(::tests::get_native_backend())
        }
    }
}

mod activation;
mod convolutional;
mod softmax;
mod pooling;

mod bench_all;
