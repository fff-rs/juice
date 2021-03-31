use crate::co::backend::{Backend, IBackend};
use crate::co::frameworks::native::flatbox::FlatBox;
use crate::co::frameworks::{Cuda, Native};
use crate::co::tensor::SharedTensor;
use env_logger;

pub fn test_setup() {
    let _ = env_logger::builder().is_test(true).try_init();
}

pub fn test_teardown() {}

pub fn get_native_backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}

pub fn get_cuda_backend() -> Backend<Cuda> {
    Backend::<Cuda>::default().unwrap()
}

pub fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
    let mem_buffer = mem.as_mut_slice::<T>();
    for (index, datum) in data.iter().enumerate() {
        mem_buffer[index] = *datum;
    }
}

pub fn filled_tensor<B: IBackend, T: Copy>(_backend: &B, n: usize, val: T) -> SharedTensor<T> {
    let mut x = SharedTensor::<T>::new(&vec![n]);
    let values: &[T] = &::std::iter::repeat(val)
        .take(x.capacity())
        .collect::<Vec<T>>();
    write_to_memory(x.write_only(get_native_backend().device()).unwrap(), values);
    x
}
