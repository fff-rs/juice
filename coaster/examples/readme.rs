use coaster as co;
use coaster_nn as nn;
#[cfg(feature = "cuda")]
use rcublas;

use nn::*;
use co::prelude::*;
use co::frameworks::native::flatbox::FlatBox;
#[cfg(feature = "cuda")]
use co::frameworks::cuda::get_cuda_backend;
#[cfg(not(feature = "cuda"))]
use co::frameworks::native::get_native_backend;

fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
    let mem_buffer = mem.as_mut_slice::<T>();
    for (index, datum) in data.iter().enumerate() {
        mem_buffer[index] = *datum;
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    let backend = get_cuda_backend();

    #[cfg(not(feature = "cuda"))]
    let backend = get_native_backend();

    // Initialize two SharedTensors.
    let mut x = SharedTensor::<f32>::new(&(1, 1, 3));
    // let mut result = SharedTensor::<f32>::new(&(1, 1, 3));
    // Fill `x` with some data.
    let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
    let native = Backend::<Native>::default().unwrap();
    write_to_memory(x.write_only(native.device()).unwrap(), payload); // Write to native host memory.
    // Run the sigmoid operation, provided by the NN Plugin, on your CUDA enabled GPU.
    // FIXME: Sigmoid cannot be included from coaster-nn without using cuda and native features
    // from coaster-nn. This causes the error https://github.com/rust-lang/cargo/issues/6915 ,
    // and so sigmoid has been disabled for now.
    // backend.sigmoid(&mut x, &mut result).unwrap();
    // See the result.
    // println!("{:?}", result.read(native.device()).unwrap().as_slice::<f32>());
}
