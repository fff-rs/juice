extern crate coaster as co;
extern crate coaster_nn as nn;

use co::prelude::*;
use co::frameworks::native::flatbox::FlatBox;
use nn::*;

fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
    let mem_buffer = mem.as_mut_slice::<T>();
    for (index, datum) in data.iter().enumerate() {
        mem_buffer[index] = *datum;
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    let backend = Backend::<Native>::default().unwrap();
    // Initialize two SharedTensors.
    let mut x = SharedTensor::<f32>::new(&(1, 1, 3));
    let mut result = SharedTensor::<f32>::new(&(1, 1, 3));
    let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
    let native = Backend::<Native>::default().unwrap();
    write_to_memory(x.write_only(native.device()).unwrap(), payload); // Write to native host memory.
    // FIXME: Add in Sigmoid example for native
}

#[cfg(feature = "cuda")]
fn main() {
    // Initialize a CUDA Backend.
    let framework = Cuda::new();
    let hardwares = framework.hardwares()[0..1].to_vec();
    let backend_config = BackendConfig::new(framework, &hardwares);
    let mut backend = Backend::new(backend_config).unwrap();
    backend.framework.initialise_cudnn().unwrap();
    // Initialize two SharedTensors.
    let mut x = SharedTensor::<f32>::new(&(1, 1, 3));
    let mut result = SharedTensor::<f32>::new(&(1, 1, 3));
    // Fill `x` with some data.
    let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
    let native = Backend::<Native>::default().unwrap();
    write_to_memory(x.write_only(native.device()).unwrap(), payload); // Write to native host memory.
    // Run the sigmoid operation, provided by the NN Plugin, on your CUDA enabled GPU.
    //backend.sigmoid(&mut x, &mut result).unwrap();
    // FIXME: Sigmoid not implemented for backend?!
    // See the result.
    println!("{:?}", result.read(native.device()).unwrap().as_slice::<f32>());
}
