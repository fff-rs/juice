extern crate coaster as co;
extern crate coaster_nn as nn;
use co::prelude::*;
use nn::*;

fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
	let mut mem_buffer = mem.as_mut_slice::<T>();
	for (index, datum) in data.iter().enumerate() {
	    mem_buffer[index] = *datum;
	}
}

fn main() {
    // Initialize a CUDA Backend.
    let backend = Backend::<Native>::default().unwrap();
    // Initialize two SharedTensors.
    let mut x = SharedTensor::<f32>::new(&(1, 1, 3));
    let mut result = SharedTensor::<f32>::new(&(1, 1, 3));
    // Fill `x` with some data.
    let payload: &[f32] = &::std::iter::repeat(1f32).take(x.capacity()).collect::<Vec<f32>>();
    let native = Backend::<Native>::default().unwrap();
    write_to_memory(x.write_only(native.device()).unwrap(), payload); // Write to native host memory.
    // Run the sigmoid operation, provided by the NN Plugin, on your CUDA enabled GPU.
    backend.sigmoid(&mut x, &mut result).unwrap();
    // See the result.
    println!("{:?}", result.read(native.device()).unwrap().as_slice::<f32>());
}
