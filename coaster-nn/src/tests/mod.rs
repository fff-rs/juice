//! This module defines common functions used by tests.
//! Test vectors for ReLU, Sigmoid, Tanh, Softmax, LogSoftmax were generated
//! by [nn-test-vectors][nn-test-vectors] script. It uses arbitrary precision
//! floating point library, so if there is need to increase precision for f128,
//! it can be easily done.
//!
//! [nn-test-vectors]: https://github.com/spearow/coaster

use std;
use std::fmt;

use rand::{thread_rng, Rng};

use crate::co::prelude::*;
use crate::co::plugin::numeric_helpers::{cast, NumCast};

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

pub fn write_to_tensor<T,F>(_backend: &Backend<F>, xs: &mut SharedTensor<T>, data: &[f64])
    where T: ::std::marker::Copy + NumCast,
          F: IFramework,
          Backend<F>: IBackend {

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
  // not functional since, PartialEq has yet to be implemented for Device
  // but tbh this is test only so screw the extra dangling ununsed memory alloc
  //       let other_dev = backend.device();
  //       if other_dev != native_dev {
  //           xs.read(other_dev).unwrap();
  //           xs.drop_device(native_dev).unwrap();
  //       }
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
pub fn uniformly_random_tensor<T,F>(_backend: &Backend<F>, dims: &[usize], low: T, high: T) -> SharedTensor<T>
    where T: Copy + PartialEq + PartialOrd + rand::distributions::uniform::SampleUniform,
          F: IFramework,
          Backend<F>: IBackend {

    let mut xs = SharedTensor::new(&dims);
    {
        let native = get_native_backend();
        let native_dev = native.device();
        {
            let mem = xs.write_only(native_dev).unwrap();
            let mem_slice = mem.as_mut_slice::<T>();

            let mut rng = thread_rng();
            for x in mem_slice {
                *x = Rng::gen_range(&mut rng, low, high);
            }
        }
  // not functional since, PartialEq has yet to be implemented for Device
  // but tbh this is test only so screw the extra dangling ununsed memory alloc
  //       let other_dev = backend.device();
  //       if other_dev != native_dev {
  //           xs.read(other_dev).unwrap();
  //           xs.drop_device(native_dev).unwrap();
  //       }
    }
    xs
}

/// This function tests that contents of `xs` and `data` don't differ too much.
/// They are allowed to differ by no more than `x * T::epsilon() * epsilon_mul`.
/// Of course if there were inevitable substantial rounding errors during
/// calculations of `xs` there may be false positives.
pub fn tensor_assert_eq<T>(xs: &SharedTensor<T>, data: &[f64], epsilon_mul: f64)
    where T: Copy + fmt::Debug + PartialEq + NumCast + Epsilon {

    let e = cast::<_, f64>(T::epsilon()).unwrap() * epsilon_mul;

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
            println!("Results differ: {:?} != {:?} ({:.2?} in {:?} and {:?}",
                     x1_t, x2, diff / max_diff, mem_slice, data);
            assert!(false);
        }
    }
}

pub fn tensor_assert_eq_tensor<T,U>(xa: &SharedTensor<T>, xb: &SharedTensor<U>, epsilon_mul: f64)
    where T: Copy + fmt::Debug + PartialEq + NumCast + Epsilon,
          U: Copy + fmt::Debug + PartialEq + NumCast + Epsilon {

    let e = cast::<_, f64>(T::epsilon()).unwrap() * epsilon_mul;

    let native = get_native_backend();
    let native_dev = native.device();

    let mem_a = xa.read(native_dev).unwrap();
    let mem_slice_a = mem_a.as_slice::<T>();

    let mem_b = xb.read(native_dev).unwrap();
    let mem_slice_b = mem_b.as_slice::<U>();

    assert_eq!(mem_slice_a.len(), mem_slice_b.len());
    for (x1, x2) in mem_slice_a.iter().zip(mem_slice_b.iter()) {
        let x1_t = cast::<_, f64>(*x1).unwrap();
        let x2_t = cast::<_, f64>(*x2).unwrap();
        let diff = (x1_t - x2_t).abs();
        let max_diff = e * (x1_t.abs() + x2_t.abs()) * 0.5;
        if (x1_t - x2_t).abs() > e * (x1_t.abs() + x2_t.abs()) * 0.5 {
            println!("Results differ: {:?} != {:?} ({:.2?} in {:?} and {:?}",
                     x1_t, x2_t, diff / max_diff, mem_slice_a, mem_slice_b);
            assert!(false);
        }
    }
}

pub fn tensor_assert_ne_tensor<T,U>(xa: &SharedTensor<T>, xb: &SharedTensor<U>, epsilon_mul: f64)
    where T: Copy + fmt::Debug + PartialEq + NumCast + Epsilon,
          U: Copy + fmt::Debug + PartialEq + NumCast + Epsilon {

    let e = cast::<_, f64>(T::epsilon()).unwrap() * epsilon_mul;

    let native = get_native_backend();
    let native_dev = native.device();

    let mem_a = xa.read(native_dev).unwrap();
    let mem_slice_a = mem_a.as_slice::<T>();

    let mem_b = xb.read(native_dev).unwrap();
    let mem_slice_b = mem_b.as_slice::<U>();

    assert_eq!(mem_slice_a.len(), mem_slice_b.len());
    for (x1, x2) in mem_slice_a.iter().zip(mem_slice_b.iter()) {
        let x1_t = cast::<_, f64>(*x1).unwrap();
        let x2_t = cast::<_, f64>(*x2).unwrap();
        if (x1_t - x2_t).abs() > e * (x1_t.abs() + x2_t.abs()) * 0.5 {
            return;
        }
    }
    println!("Results are too similar {:?} ~= {:?}", mem_slice_a, mem_slice_b);
    assert!(false);
}

// All operations for Cuda and Native are provided for f32 and f64.
// Those macros remove boilerplate in test definitions.
// concat_idents! is behind feature gate at the moment, otherwise
// invocations could be made much less verbose.
macro_rules! test_cuda {
    ($test_name:ident, $f32_name:ident, $f64_name:ident) => {

        #[cfg(feature = "cuda")]
        #[test]
        #[serial_test::serial]
        fn $f32_name() {
            $test_name::<f32, _>(crate::tests::get_cuda_backend())
        }

        #[cfg(feature = "cuda")]
        #[test]
        #[serial_test::serial]
        fn $f64_name() {
            $test_name::<f64, _>(crate::tests::get_cuda_backend())
        }
    }
}

macro_rules! test_native {
    ($test_name:ident, $f32_name:ident, $f64_name:ident) => {

        #[cfg(feature = "native")]
        #[test]
        fn $f32_name() {
            $test_name::<f32, _>(crate::tests::get_native_backend())
        }

        #[cfg(feature = "native")]
        #[test]
        fn $f64_name() {
            $test_name::<f64, _>(crate::tests::get_native_backend())
        }
    }
}

macro_rules! test_cross {
    ($test_name:ident, $f32_name:ident) => {
        #[cfg(all(feature = "native",feature = "cuda"))]
        #[test]
        #[serial_test::serial]
        fn $f32_name() {
            $test_name::<_, _>(crate::tests::get_native_backend(), crate::tests::get_cuda_backend())
        }
    }
}

mod activation;
mod convolutional;
mod softmax;
mod pooling;
mod dropout;
mod bench_all;
