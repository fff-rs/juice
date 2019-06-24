#![feature(test)]

extern crate test;
extern crate coaster as co;
extern crate coaster_blas as co_blas;
extern crate rblas;
extern crate rand;

use test::Bencher;
use crate::co::prelude::*;
use crate::co_blas::plugin::*;

use rand::{thread_rng, Rng};

fn backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}

fn bench_dot_rblas(b: &mut Bencher, n: usize) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(n).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(n).collect::<Vec<f32>>();

    b.iter(|| {
        let res = rblas::Dot::dot(&slice_a, &slice_b);
        test::black_box(res);
    });
}

fn bench_dot_coaster(b: &mut Bencher, n: usize) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(n).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(n).collect::<Vec<f32>>();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(&[n]);
    let shared_b = &mut SharedTensor::<f32>::new(&[n]);
    let shared_res = &mut SharedTensor::<f32>::new(&[1]);
    shared_a.write_only(backend.device()).unwrap()
        .as_mut_slice().clone_from_slice(&slice_a);
    shared_b.write_only(backend.device()).unwrap()
        .as_mut_slice().clone_from_slice(&slice_b);
    let _ = backend.dot(shared_a, shared_b, shared_res);

    b.iter(|| backend.dot(shared_a, shared_b, shared_res).unwrap());
}



#[bench]
fn bench_dot_100_rblas(b: &mut Bencher) { bench_dot_rblas(b, 100); }

#[bench]
fn bench_dot_100_coaster(b: &mut Bencher) { bench_dot_coaster(b, 100); }

#[bench]
fn bench_dot_1000_rblas(b: &mut Bencher) { bench_dot_rblas(b, 1000); }

#[bench]
fn bench_dot_1000_coaster(b: &mut Bencher) { bench_dot_coaster(b, 1000); }

#[bench]
fn bench_dot_2000_rblas(b: &mut Bencher) { bench_dot_rblas(b, 2000); }

#[bench]
fn bench_dot_2000_coaster(b: &mut Bencher) { bench_dot_coaster(b, 2000); }

#[bench]
fn bench_dot_10000_rblas(b: &mut Bencher) { bench_dot_rblas(b, 10000); }

#[bench]
fn bench_dot_10000_coaster(b: &mut Bencher) { bench_dot_coaster(b, 10000); }

#[bench]
fn bench_dot_20000_rblas(b: &mut Bencher) { bench_dot_rblas(b, 20000); }

#[bench]
fn bench_dot_20000_coaster(b: &mut Bencher) { bench_dot_coaster(b, 20000); }
