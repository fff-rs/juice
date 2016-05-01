#![cfg(feature = "unstable")]
extern crate test;

macro_rules! bench_activation {
    ($backend_getter:ident, $t:ident, $f:ident, $f_grad:ident,
     $bench_name:ident, $bench_grad_name:ident, $n:expr) => {
        #[bench]
        pub fn $bench_name(b: &mut Bencher) {
            let backend = ::tests::$backend_getter();

            let x = ::tests::uniformly_random_tensor(&[$n], -2.0, 2.0);
            let mut r = SharedTensor::<$t>::new(&[$n]);

            for _ in 0..3 { // warmup
                backend.$f(&x, &mut r).unwrap();
                backend.synchronize().unwrap();
            }

            b.iter(|| {
                backend.$f(&x, &mut r).unwrap();
                backend.synchronize().unwrap();
            });
        }

        #[bench]
        pub fn $bench_grad_name(b: &mut Bencher) {
            let backend = ::tests::$backend_getter();

            let mut x  = SharedTensor::<$t>::new(&[$n]);
            let dx = ::tests::uniformly_random_tensor(&[$n], -2.0, 2.0);
            let r  = ::tests::uniformly_random_tensor(&[$n], -2.0, 2.0);
            let mut dr = SharedTensor::<$t>::new(&[$n]);

            backend.$f(&r, &mut x).unwrap();

            for _ in 0..3 { // warmup
                backend.$f_grad(&x, &dx, &r, &mut dr).unwrap();
                backend.synchronize().unwrap();
            }

            b.iter(|| {
                backend.$f_grad(&x, &dx, &r, &mut dr).unwrap();
                backend.synchronize().unwrap();
            });
        }
    }
}

// softmax differs from activations only in arg count for grad function...
macro_rules! bench_softmax {
    ($backend_getter:ident, $t:ident, $f:ident, $f_grad:ident,
     $bench_name:ident, $bench_grad_name:ident, $n:expr) => {
        #[bench]
        pub fn $bench_name(b: &mut Bencher) {
            let backend = ::tests::$backend_getter();

            let x = ::tests::uniformly_random_tensor(&[$n], -2.0, 2.0);
            let mut r = SharedTensor::<$t>::new(&[$n]);

            for _ in 0..3 { // warmup
                backend.$f(&x, &mut r).unwrap();
                backend.synchronize().unwrap();
            }

            b.iter(|| {
                backend.$f(&x, &mut r).unwrap();
                backend.synchronize().unwrap();
            });
        }

        #[bench]
        pub fn $bench_grad_name(b: &mut Bencher) {
            let backend = ::tests::$backend_getter();

            let mut x  = SharedTensor::<$t>::new(&[$n]);
            let dx = ::tests::uniformly_random_tensor(&[$n], -2.0, 2.0);
            let r  = ::tests::uniformly_random_tensor(&[$n], -2.0, 2.0);
            let mut dr = SharedTensor::<$t>::new(&[$n]);

            backend.$f(&r, &mut x).unwrap();

            for _ in 0..3 { // warmup
                backend.$f_grad(&x, &dx, &mut dr).unwrap();
                backend.synchronize().unwrap();
            }

            b.iter(|| {
                backend.$f_grad(&x, &dx, &mut dr).unwrap();
                backend.synchronize().unwrap();
            });
        }
    }
}

macro_rules! define_benches { ($b:ident, $t:ident) => {
    use super::test::Bencher;
    use co::prelude::*;
    use plugin::{Relu, Sigmoid, Tanh, Softmax, LogSoftmax};

    bench_activation!($b, $t, relu, relu_grad, relu_100, relu_grad_100, 100);
    bench_activation!($b, $t, relu, relu_grad, relu_1k, relu_grad_1k, 1000);
    bench_activation!($b, $t, relu, relu_grad, relu_10k, relu_grad_10k, 10_000);
    bench_activation!($b, $t, relu, relu_grad, relu_100k, relu_grad_100k, 100_000);
    bench_activation!($b, $t, relu, relu_grad, relu_1m, relu_grad_1m, 1000_000);
    bench_activation!($b, $t, relu, relu_grad, relu_10m, relu_grad_10m, 10_000_000);

    bench_activation!($b, $t, sigmoid, sigmoid_grad, sigmoid_100, sigmoid_grad_100, 100);
    bench_activation!($b, $t, sigmoid, sigmoid_grad, sigmoid_1k, sigmoid_grad_1k, 1000);
    bench_activation!($b, $t, sigmoid, sigmoid_grad, sigmoid_10k, sigmoid_grad_10k, 10_000);
    bench_activation!($b, $t, sigmoid, sigmoid_grad, sigmoid_100k, sigmoid_grad_100k, 100_000);
    bench_activation!($b, $t, sigmoid, sigmoid_grad, sigmoid_1m, sigmoid_grad_1m, 1000_000);
    bench_activation!($b, $t, sigmoid, sigmoid_grad, sigmoid_10m, sigmoid_grad_10m, 10_000_000);

    bench_activation!($b, $t, tanh, tanh_grad, tanh_100, tanh_grad_100, 100);
    bench_activation!($b, $t, tanh, tanh_grad, tanh_1k, tanh_grad_1k, 1000);
    bench_activation!($b, $t, tanh, tanh_grad, tanh_10k, tanh_grad_10k, 10_000);
    bench_activation!($b, $t, tanh, tanh_grad, tanh_100k, tanh_grad_100k, 100_000);
    bench_activation!($b, $t, tanh, tanh_grad, tanh_1m, tanh_grad_1m, 1000_000);
    bench_activation!($b, $t, tanh, tanh_grad, tanh_10m, tanh_grad_10m, 10_000_000);

    bench_softmax!($b, $t, softmax, softmax_grad, softmax_10, softmax_grad_10, 10);
    bench_softmax!($b, $t, softmax, softmax_grad, softmax_100, softmax_grad_100, 100);
    bench_softmax!($b, $t, softmax, softmax_grad, softmax_1k, softmax_grad_1k, 1000);
    bench_softmax!($b, $t, softmax, softmax_grad, softmax_10k, softmax_grad_10k, 10_000);

    bench_softmax!($b, $t, log_softmax, log_softmax_grad,
                   log_softmax_10, log_softmax_grad_10, 10);
    bench_softmax!($b, $t, log_softmax, log_softmax_grad,
                   log_softmax_100, log_softmax_grad_100, 100);
    bench_softmax!($b, $t, log_softmax, log_softmax_grad,
                   log_softmax_1k, log_softmax_grad_1k, 1000);
    bench_softmax!($b, $t, log_softmax, log_softmax_grad,
                   log_softmax_10k, log_softmax_grad_10k, 10_000);
}}


mod native_f32 { define_benches!(get_native_backend, f32); }
mod native_f64 { define_benches!(get_native_backend, f64); }

#[cfg(feature = "cuda")]
mod cuda_f32 { define_benches!(get_cuda_backend, f32); }
#[cfg(feature = "cuda")]
mod cuda_f64 { define_benches!(get_cuda_backend, f64); }
