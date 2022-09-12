#![allow(unused_crate_dependencies, dead_code, warnings)]

use paste::paste;

use std::mem::transmute;

use crate::co::SharedTensor;
use crate::plugin::*;
use crate::tests::*;
use crate::*;
use criterion::Criterion;
use rand::distributions::OpenClosed01;

macro_rules! bench_pooling {
    ($b:ident, $t:ident, $f:ident, $n:literal) => {
        paste! {
        fn [< $f _ $n _ $b:lower _ $t:lower>](_: &mut Criterion) {
            println!("TODO FIXME");
        }

        fn [< $f _grad_ $n _ $b:lower _ $t:lower>](_: &mut Criterion) {

            println!("TODO FIXME");
        }}
    };
}

macro_rules! bench_activation {
    ($b:ident, $t:ident, $f:ident, $n:literal) => {
        paste! {
        pub fn [< $f _ $n _ $b:lower _ $t:lower>] (criterion: &mut ::criterion::Criterion) {
            let backend = Backend::<$b>::default().unwrap();
            let x = uniformly_random_tensor(&backend, &[$n], -2.0, 2.0);
            let mut r = SharedTensor::<$t>::new(&[$n]);

            for _ in 0..3 {
                // warmup
                backend. $f (&x, &mut r).unwrap();
                backend.synchronize().unwrap();
            }

            criterion.bench_function(stringify!($f), |b| {
                b.iter(|| {
                    backend. $f (&x, &mut r).unwrap();
                    backend.synchronize().unwrap();
                })
            });
        }

        pub fn [< $f _grad_ $n _ $b:lower _ $t:lower>] (criterion: &mut ::criterion::Criterion) {
            let backend = Backend::<$b>::default().unwrap();

            let mut x = SharedTensor::<$t>::new(&[$n]);
            let dx = uniformly_random_tensor(&backend, &[$n], -2.0, 2.0);
            let r = uniformly_random_tensor(&backend, &[$n], -2.0, 2.0);
            let mut dr = SharedTensor::<$t>::new(&[$n]);

            backend. $f (&r, &mut x).unwrap();

            for _ in 0..3 {
                // warmup
                backend.[<$f _grad>] (&x, &dx, &r, &mut dr).unwrap();
                backend.synchronize().unwrap();
            }

            criterion.bench_function(stringify!($f_grad), |b| {
                b.iter(|| {
                    backend.[< $f _grad >] (&x, &dx, &r, &mut dr).unwrap();
                    backend.synchronize().unwrap();
                })
            });
        }}
    };
}

// softmax differs from activations only in arg count for grad function...
macro_rules! bench_softmax {
    ($b:ident, $t:ident, $f:ident,
     $n:literal) => {
        use super::*;
        use crate::plugin::*;

        paste!{
        pub fn [< $f _ $n _ $b:lower _ $t:lower>](criterion: &mut ::criterion::Criterion) {
            let backend = Backend::<$b>::default().unwrap();

            let x = uniformly_random_tensor(&backend, &[$n], -2.0, 2.0);
            let mut r = SharedTensor::<$t>::new(&[$n]);

            for _ in 0..3 {
                // warmup
                backend. $f (&x, &mut r).unwrap();
                backend.synchronize().unwrap();
            }

            criterion.bench_function(stringify!($bench_name), |b| {
                b.iter(|| {
                    backend.$f(&x, &mut r).unwrap();
                    backend.synchronize().unwrap();
                })
            });
        }

        pub fn [< $f _grad_ $n _ $b:lower _ $t:lower>] (criterion: &mut ::criterion::Criterion) {
            let backend = Backend::<$b>::default().unwrap();

            let mut x = SharedTensor::<$t>::new(&[$n]);
            let dx = uniformly_random_tensor(&backend, &[$n], -2.0, 2.0);
            let r = uniformly_random_tensor(&backend, &[$n], -2.0, 2.0);
            let mut dr = SharedTensor::<$t>::new(&[$n]);

            backend. $f (&r, &mut x).unwrap();

            for _ in 0..3 {
                // warmup
                backend. [<$f _grad>] (&x, &dx, &mut dr).unwrap();
                backend.synchronize().unwrap();
            }

            criterion.bench_function(stringify!([< $f _grad _ $n _ $b:lower _ $t:lower >]), |b| {
                b.iter(|| {
                    backend. [<$f _grad>] (&x, &dx, &mut dr).unwrap();
                    backend.synchronize().unwrap();
                })
            });
        }
    }
    };
}

macro_rules! add_bench_group {
    ($b:ident, $t:ident, $f:ident, $k:ident, [$($n:literal),+], $criterion:expr) => {

        paste!{
            $(
                $k !($b,$t,$f,$n);
            )+
            $({
                let fx = [< $f _ $n _ $b:lower _ $t:lower>];
                fx($criterion);

                let fx_grad = [< $f _grad_ $n _ $b:lower _ $t:lower >];
                fx_grad($criterion);
            })+
        }
    }
}

use super::*;
use co::prelude::*;
use plugin::{LogSoftmax, Relu, Sigmoid, Softmax, Tanh};

macro_rules! define_benches {
    ($b:ident, $t:ident) => {
        paste!{
            fn [<group_ $b:lower _ $t:lower>]() {
                let mut criterion = ::criterion::Criterion::default()
                .configure_from_args();
                add_bench_group!($b, $t, relu, bench_activation, [100,1000,10_000,100_000,1_000_000, 10_000_000], &mut criterion);
                add_bench_group!($b, $t, sigmoid, bench_activation, [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000], &mut criterion);
                add_bench_group!($b, $t, tanh, bench_activation, [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000], &mut criterion);
                add_bench_group!($b, $t, softmax, bench_softmax, [10, 100, 1_000], &mut criterion);
                add_bench_group!($b, $t, log_softmax, bench_softmax,[10, 100, 1_000, 10_000], &mut criterion);
                add_bench_group!($b, $t, pooling_avg, bench_pooling, [10, 100, 1_000, 10_000], &mut criterion);
                add_bench_group!($b, $t, pooling_max, bench_pooling, [10, 100, 1_000, 10_000], &mut criterion);
            }
        }
    }
}

define_benches!(Native, f32);
define_benches!(Native, f64);

#[cfg(feature = "cuda")]
define_benches!(Cuda, f32);
#[cfg(feature = "cuda")]
define_benches!(Cuda, f64);

fn main() {
    group_cuda_f32();
    group_cuda_f64();
    group_native_f32();
    group_native_f64();
    ::criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
