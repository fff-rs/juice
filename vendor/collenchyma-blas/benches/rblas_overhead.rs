#![feature(test)]
#![feature(clone_from_slice)]

extern crate test;
extern crate collenchyma as co;
extern crate collenchyma_blas as co_blas;
extern crate rblas;
extern crate rand;

use test::Bencher;
use co::backend::{Backend, BackendConfig};
use co::frameworks::Native;
use co::framework::IFramework;
use co::tensor::SharedTensor;
use co_blas::plugin::*;
use rblas::Dot;

use rand::{thread_rng, Rng};

fn backend() -> Backend<Native> {
    let framework = Native::new();
    let hardwares = framework.hardwares();
    let backend_config = BackendConfig::new(framework, hardwares);
    Backend::new(backend_config).unwrap()
}

#[bench]
fn bench_1000_dot_100_rblas(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(100).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(100).collect::<Vec<f32>>();

    b.iter(|| {
        for _ in 0..1000 {
            let res = Dot::dot(&slice_a, &slice_b);
            test::black_box(res);
        }
    });
}

#[bench]
fn bench_1000_dot_100_collenchyma(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(100).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(100).collect::<Vec<f32>>();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(backend.device(), &100).unwrap();
    let shared_b = &mut SharedTensor::<f32>::new(backend.device(), &100).unwrap();
    let shared_res = &mut SharedTensor::<f32>::new(backend.device(), &()).unwrap();
    shared_a.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_a);
    shared_b.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_b);
    let _ = backend.dot(shared_a, shared_b, shared_res);
    bench_1000_dot_100_collenchyma_profile(b, &backend, shared_a, shared_b, shared_res);
}

#[inline(never)]
fn bench_1000_dot_100_collenchyma_profile(
    b: &mut Bencher,
    backend: &Backend<Native>,
    shared_a: &mut SharedTensor<f32>,
    shared_b: &mut SharedTensor<f32>,
    shared_res: &mut SharedTensor<f32>
) {
    b.iter(|| {
        for _ in 0..1000 {
            let _ = backend.dot(shared_a, shared_b, shared_res);
        }
    });
}

#[bench]
fn bench_1000_dot_100_collenchyma_plain(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(100).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(100).collect::<Vec<f32>>();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(backend.device(), &100).unwrap();
    let shared_b = &mut SharedTensor::<f32>::new(backend.device(), &100).unwrap();
    let shared_res = &mut SharedTensor::<f32>::new(backend.device(), &()).unwrap();
    shared_a.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_a);
    shared_b.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_b);
    let _ = backend.dot(shared_a, shared_b, shared_res);
    bench_1000_dot_100_collenchyma_plain_profile(b, &backend, shared_a, shared_b, shared_res);
}

#[inline(never)]
fn bench_1000_dot_100_collenchyma_plain_profile(
    b: &mut Bencher,
    backend: &Backend<Native>,
    shared_a: &mut SharedTensor<f32>,
    shared_b: &mut SharedTensor<f32>,
    shared_res: &mut SharedTensor<f32>
) {
    b.iter(|| {
        for _ in 0..1000 {
            let _ = backend.dot_plain(shared_a, shared_b, shared_res);
        }
    });
}

#[bench]
fn bench_100_dot_1000_rblas(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(1000).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(1000).collect::<Vec<f32>>();

    b.iter(|| {
        for _ in 0..100 {
            let res = Dot::dot(&slice_a, &slice_b);
            test::black_box(res);
        }
    });
}

#[bench]
fn bench_100_dot_1000_collenchyma(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(1000).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(1000).collect::<Vec<f32>>();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(backend.device(), &1000).unwrap();
    let shared_b = &mut SharedTensor::<f32>::new(backend.device(), &1000).unwrap();
    let shared_res = &mut SharedTensor::<f32>::new(backend.device(), &()).unwrap();
    shared_a.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_a);
    shared_b.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_b);
    let _ = backend.dot(shared_a, shared_b, shared_res);
    bench_100_dot_1000_collenchyma_profile(b, &backend, shared_a, shared_b, shared_res);
}

#[inline(never)]
fn bench_100_dot_1000_collenchyma_profile(
    b: &mut Bencher,
    backend: &Backend<Native>,
    shared_a: &mut SharedTensor<f32>,
    shared_b: &mut SharedTensor<f32>,
    shared_res: &mut SharedTensor<f32>
) {
    b.iter(|| {
        for _ in 0..100 {
            let _ = backend.dot(shared_a, shared_b, shared_res);
        }
    });
}

#[bench]
fn bench_50_dot_2000_collenchyma(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(2000).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(2000).collect::<Vec<f32>>();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(backend.device(), &2000).unwrap();
    let shared_b = &mut SharedTensor::<f32>::new(backend.device(), &2000).unwrap();
    let shared_res = &mut SharedTensor::<f32>::new(backend.device(), &()).unwrap();
    shared_a.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_a);
    shared_b.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_b);
    let _ = backend.dot(shared_a, shared_b, shared_res);
    bench_50_dot_2000_collenchyma_profile(b, &backend, shared_a, shared_b, shared_res);
}

#[inline(never)]
fn bench_50_dot_2000_collenchyma_profile(
    b: &mut Bencher,
    backend: &Backend<Native>,
    shared_a: &mut SharedTensor<f32>,
    shared_b: &mut SharedTensor<f32>,
    shared_res: &mut SharedTensor<f32>
) {
    b.iter(|| {
        for _ in 0..50 {
            let _ = backend.dot(shared_a, shared_b, shared_res);
        }
    });
}

#[bench]
fn bench_10_dot_10000_rblas(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(10000).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(10000).collect::<Vec<f32>>();

    b.iter(|| {
        for _ in 0..10 {
            let res = Dot::dot(&slice_a, &slice_b);
            test::black_box(res);
        }
    });
}

#[bench]
fn bench_10_dot_10000_collenchyma(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(10000).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(10000).collect::<Vec<f32>>();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(backend.device(), &10000).unwrap();
    let shared_b = &mut SharedTensor::<f32>::new(backend.device(), &10000).unwrap();
    let shared_res = &mut SharedTensor::<f32>::new(backend.device(), &()).unwrap();
    shared_a.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_a);
    shared_b.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_b);
    let _ = backend.dot(shared_a, shared_b, shared_res);
    bench_10_dot_10000_collenchyma_profile(b, &backend, shared_a, shared_b, shared_res);
}

#[inline(never)]
fn bench_10_dot_10000_collenchyma_profile(
    b: &mut Bencher,
    backend: &Backend<Native>,
    shared_a: &mut SharedTensor<f32>,
    shared_b: &mut SharedTensor<f32>,
    shared_res: &mut SharedTensor<f32>
) {
    b.iter(|| {
        for _ in 0..10 {
            let _ = backend.dot(shared_a, shared_b, shared_res);
        }
    });
}

#[bench]
fn bench_5_dot_20000_rblas(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(20000).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(20000).collect::<Vec<f32>>();

    b.iter(|| {
        for _ in 0..5 {
            let res = Dot::dot(&slice_a, &slice_b);
            test::black_box(res);
        }
    });
}

#[bench]
fn bench_5_dot_20000_collenchyma(b: &mut Bencher) {
    let mut rng = thread_rng();
    let slice_a = rng.gen_iter::<f32>().take(20000).collect::<Vec<f32>>();
    let slice_b = rng.gen_iter::<f32>().take(20000).collect::<Vec<f32>>();

    let backend = backend();
    let shared_a = &mut SharedTensor::<f32>::new(backend.device(), &20000).unwrap();
    let shared_b = &mut SharedTensor::<f32>::new(backend.device(), &20000).unwrap();
    let shared_res = &mut SharedTensor::<f32>::new(backend.device(), &()).unwrap();
    shared_a.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_a);
    shared_b.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_b);
    let _ = backend.dot(shared_a, shared_b, shared_res);
    bench_5_dot_20000_collenchyma_profile(b, &backend, shared_a, shared_b, shared_res);
}

#[inline(never)]
fn bench_5_dot_20000_collenchyma_profile(
    b: &mut Bencher,
    backend: &Backend<Native>,
    shared_a: &mut SharedTensor<f32>,
    shared_b: &mut SharedTensor<f32>,
    shared_res: &mut SharedTensor<f32>
) {
    b.iter(|| {
        for _ in 0..5 {
            let _ = backend.dot(shared_a, shared_b, shared_res);
        }
    });
}
