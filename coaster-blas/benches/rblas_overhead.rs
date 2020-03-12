#![feature(test)]

extern crate coaster as co;
extern crate coaster_blas as co_blas;
extern crate rand;
extern crate rust_blas as rblas;
extern crate test;

use crate::co::prelude::*;
use crate::co_blas::plugin::*;
use test::Bencher;

use rand::distributions::Standard;
use rand::{thread_rng, Rng};

fn backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}

fn bench_dot_rblas(b: &mut Bencher, n: usize) {
    let rng = thread_rng();
    let slice_a: Vec<f32> = rng.sample_iter(Standard).take(n).collect();
    let slice_b: Vec<f32> = rng.sample_iter(Standard).take(n).collect();

    b.iter(|| {
        let res = rblas::Dot::dot(slice_a.as_slice(), slice_b.as_slice());
        test::black_box(res);
    });
}

fn bench_dot_coaster(b: &mut Bencher, n: usize) {
    let rng = thread_rng();
    let slice_a: Vec<f32> = rng.sample_iter(Standard).take(n).collect();
    let slice_b: Vec<f32> = rng.sample_iter(Standard).take(n).collect();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(&[n]);
    let shared_b = &mut SharedTensor::<f32>::new(&[n]);
    let shared_res = &mut SharedTensor::<f32>::new(&[1]);
    shared_a
        .write_only(backend.device())
        .unwrap()
        .as_mut_slice()
        .clone_from_slice(slice_a.as_slice());
    shared_b
        .write_only(backend.device())
        .unwrap()
        .as_mut_slice()
        .clone_from_slice(slice_b.as_slice());
    let _ = backend.dot(shared_a, shared_b, shared_res);

    b.iter(|| backend.dot(shared_a, shared_b, shared_res).unwrap());
}

#[bench]
fn bench_dot_100_rblas(b: &mut Bencher) {
    bench_dot_rblas(b, 100);
}

#[bench]
fn bench_dot_100_coaster(b: &mut Bencher) {
    bench_dot_coaster(b, 100);
}

#[bench]
fn bench_dot_1000_rblas(b: &mut Bencher) {
    bench_dot_rblas(b, 1000);
}

#[bench]
fn bench_dot_1000_coaster(b: &mut Bencher) {
    bench_dot_coaster(b, 1000);
}

#[bench]
fn bench_dot_2000_rblas(b: &mut Bencher) {
    bench_dot_rblas(b, 2000);
}

#[bench]
fn bench_dot_2000_coaster(b: &mut Bencher) {
    bench_dot_coaster(b, 2000);
}

#[bench]
fn bench_dot_10000_rblas(b: &mut Bencher) {
    bench_dot_rblas(b, 10000);
}

#[bench]
fn bench_dot_10000_coaster(b: &mut Bencher) {
    bench_dot_coaster(b, 10000);
}

#[bench]
fn bench_dot_20000_rblas(b: &mut Bencher) {
    bench_dot_rblas(b, 20000);
}

#[bench]
fn bench_dot_20000_coaster(b: &mut Bencher) {
    bench_dot_coaster(b, 20000);
}
