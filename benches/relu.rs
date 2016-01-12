#![feature(test)]
#![feature(clone_from_slice)]

extern crate test;
extern crate collenchyma as co;
extern crate collenchyma_nn as co_nn;
extern crate rand;

use test::Bencher;
use co::backend::{Backend, BackendConfig};
use co::frameworks::Native;
use co::framework::IFramework;
use co::tensor::SharedTensor;
use co_nn::*;

use rand::{thread_rng, Rng};

fn backend() -> Backend<Native> {
    let framework = Native::new();
    let hardwares = framework.hardwares();
    let backend_config = BackendConfig::new(framework, hardwares);
    Backend::new(backend_config).unwrap()
}

fn arguments<T: IFramework + Clone>(backend: &Backend<T>, size: usize) -> (SharedTensor<f32>, SharedTensor<f32>) {
    let mut rng = thread_rng();
    let slice_x = rng.gen_iter::<f32>().take(size).collect::<Vec<f32>>();

    let mut x = SharedTensor::<f32>::new(backend.device(), &size).unwrap();
    let out = SharedTensor::<f32>::new(backend.device(), &size).unwrap();
    x.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_x);
    (x, out)
}

fn arguments_grad<T: IFramework + Clone>(backend: &Backend<T>, size: usize) -> (SharedTensor<f32>, SharedTensor<f32>, SharedTensor<f32>, SharedTensor<f32>) {
    let mut rng = thread_rng();
    let slice_x = rng.gen_iter::<f32>().take(size).collect::<Vec<f32>>();

    let mut x = SharedTensor::<f32>::new(backend.device(), &size).unwrap();
    let mut dx = SharedTensor::<f32>::new(backend.device(), &size).unwrap();
    let mut out = SharedTensor::<f32>::new(backend.device(), &size).unwrap();
    let dout = SharedTensor::<f32>::new(backend.device(), &size).unwrap();
    x.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_x);
    dx.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_x);
    out.get_mut(backend.device()).unwrap().as_mut_native().unwrap().as_mut_slice().clone_from_slice(&slice_x);
    (x, dx, out, dout)
}

#[inline(never)]
fn bench_profile<F: FnMut() -> ()>(
    b: &mut Bencher,
    mut bench_func: F,
    times: usize
) {
    b.iter(|| { for _ in 0..times { bench_func(); } });
}

#[bench]
fn bench_1000_relu_100_native(b: &mut Bencher) {
    let backend = backend();
    let (mut x, mut out) = arguments(&backend, 100);
    let mut func = || { let _ = backend.relu_plain(&mut x, &mut out); };
    { func(); bench_profile(b, func, 1000); }
}

#[bench]
fn bench_10_relu_10000_native(b: &mut Bencher) {
    let backend = backend();
    let (mut x, mut out) = arguments(&backend, 10000);
    let mut func = || { let _ = backend.relu_plain(&mut x, &mut out); };
    { func(); bench_profile(b, func, 10); }
}

#[bench]
fn bench_1000_relu_grad_100_native(b: &mut Bencher) {
    let backend = backend();
    let (mut x, mut dx, mut out, mut dout) = arguments_grad(&backend, 100);
    let mut func = || { let _ = backend.relu_grad_plain(&mut x, &mut dx, &mut out, &mut dout); };
    { func(); bench_profile(b, func, 1000); }
}

#[bench]
fn bench_10_relu_grad_10000_native(b: &mut Bencher) {
    let backend = backend();
    let (mut x, mut dx, mut out, mut dout) = arguments_grad(&backend, 10000);
    let mut func = || { let _ = backend.relu_grad_plain(&mut x, &mut dx, &mut out, &mut dout); };
    { func(); bench_profile(b, func, 10); }
}
