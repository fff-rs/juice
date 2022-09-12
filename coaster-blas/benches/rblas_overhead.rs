use coaster as co;
use coaster_blas as co_blas;
use rust_blas as rblas;

use co::prelude::*;
use co_blas::plugin::*;
use criterion::{criterion_group, criterion_main, Criterion};

use rand::distributions::Standard;
use rand::{thread_rng, Rng};

fn backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}

fn bench_dot_rblas(b: &mut Criterion, n: usize) {
    let rng = &mut thread_rng();
    let slice_a: Vec<f32> = rng.sample_iter(Standard).take(n).collect();
    let slice_b: Vec<f32> = rng.sample_iter(Standard).take(n).collect();
    b.bench_function("bench dot rblas", |b| {
        b.iter(|| {
            let res = rblas::Dot::dot(slice_a.as_slice(), slice_b.as_slice());
            ::criterion::black_box(res);
        })
    });
}

fn bench_dot_coaster(b: &mut Criterion, n: usize) {
    let rng = &mut thread_rng();
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
    b.bench_function("bench dot coaster", |b| {
        b.iter(|| backend.dot(shared_a, shared_b, shared_res).unwrap())
    });
}

fn bench_dot_100_rblas(b: &mut Criterion) {
    bench_dot_rblas(b, 100);
}

fn bench_dot_100_coaster(b: &mut Criterion) {
    bench_dot_coaster(b, 100);
}

fn bench_dot_1000_rblas(b: &mut Criterion) {
    bench_dot_rblas(b, 1000);
}

fn bench_dot_1000_coaster(b: &mut Criterion) {
    bench_dot_coaster(b, 1000);
}

fn bench_dot_2000_rblas(b: &mut Criterion) {
    bench_dot_rblas(b, 2000);
}

fn bench_dot_2000_coaster(b: &mut Criterion) {
    bench_dot_coaster(b, 2000);
}

fn bench_dot_10000_rblas(b: &mut Criterion) {
    bench_dot_rblas(b, 10000);
}

fn bench_dot_10000_coaster(b: &mut Criterion) {
    bench_dot_coaster(b, 10000);
}

fn bench_dot_20000_rblas(b: &mut Criterion) {
    bench_dot_rblas(b, 20000);
}

fn bench_dot_20000_coaster(b: &mut Criterion) {
    bench_dot_coaster(b, 20000);
}

criterion_group!(
    coaster_nn,
    bench_dot_100_rblas,
    bench_dot_100_coaster,
    bench_dot_1000_rblas,
    bench_dot_1000_coaster,
    bench_dot_2000_rblas,
    bench_dot_2000_coaster,
    bench_dot_10000_rblas,
    bench_dot_10000_coaster,
    bench_dot_20000_rblas,
    bench_dot_20000_coaster,
);

criterion_main!(coaster_nn);
