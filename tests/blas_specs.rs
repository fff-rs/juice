extern crate collenchyma as co;
extern crate libc;
extern crate num;

#[cfg(test)]
mod blas_spec {

    use co::backend::{Backend, BackendConfig};
    use co::framework::IFramework;
    use co::frameworks::{OpenCL, Native};
    use co::libraries::blas::*;
    use co::memory::MemoryType;
    use co::shared_memory::{SharedMemory, ITensor, TensorR0, TensorR1};
    use num::traits::{cast, NumCast, Float};

    fn get_native_backend() -> Backend<Native> {
        let framework = Native::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Backend::new(backend_config).unwrap()
    }

    fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
        if let &mut MemoryType::Native(ref mut mem) = mem {
            let mut mem_buffer = mem.as_mut_slice::<T>();
            for (index, datum) in data.iter().enumerate() {
                mem_buffer[index] = *datum;
            }
        }
    }

    fn get_asum_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T, TensorR1>, SharedMemory<T, TensorR0>){
        let mut x = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(-2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let result = SharedMemory::<T, TensorR0>::new(backend.device(), TensorR0::new()).unwrap();
        (x, result)
    }

    fn get_axpy_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T, TensorR0>, SharedMemory<T, TensorR1>, SharedMemory<T, TensorR1>){
        let mut a = SharedMemory::<T, TensorR0>::new(backend.device(), TensorR0::new()).unwrap();
        write_to_memory(a.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(2).unwrap()]);

        let mut x = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);
        (a, x, y)
    }

    fn get_copy_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T, TensorR1>, SharedMemory<T, TensorR1>){
        let mut x = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let y = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        (x, y)
    }

    fn get_dot_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T, TensorR1>, SharedMemory<T, TensorR1>, SharedMemory<T, TensorR0>){
        let mut x = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let result = SharedMemory::<T, TensorR0>::new(backend.device(), TensorR0::new()).unwrap();
        (x, y, result)
    }

    fn get_nrm2_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T, TensorR1>, SharedMemory<T, TensorR0>){
        let mut x = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(2).unwrap()]);

        let result = SharedMemory::<T, TensorR0>::new(backend.device(), TensorR0::new()).unwrap();
        (x, result)
    }

    fn get_scale_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T, TensorR0>, SharedMemory<T, TensorR1>){
        let mut a = SharedMemory::<T, TensorR0>::new(backend.device(), TensorR0::new()).unwrap();
        write_to_memory(a.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(2).unwrap()]);

        let mut y = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        (a, y)
    }

    fn get_swap_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T, TensorR1>, SharedMemory<T, TensorR1>){
        let mut x = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedMemory::<T, TensorR1>::new(backend.device(), TensorR1::new([3])).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(3).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(1).unwrap()]);

        (x, y)
    }

    #[test]
    fn it_computes_correct_asum_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_asum_memory::<f32, Native>(&backend);

        if let Ok(_) = backend.asum(&mut x, &mut result) {
            if let Some(mem) = result.get(backend.device()).unwrap().as_native() { assert_eq!(&[6f32], mem.as_slice::<f32>()) }
        }
        backend.asum(&mut x, &mut result).unwrap();
    }

    #[test]
    fn it_computes_correct_asum_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_asum_memory::<f64, Native>(&backend);

        if let Ok(_) = backend.asum(&mut x, &mut result) {
            if let Some(mem) = result.get(backend.device()).unwrap().as_native() { assert_eq!(&[6f64], mem.as_slice::<f64>()) }
        }
        backend.asum(&mut x, &mut result).unwrap();
    }

    #[test]
    fn it_computes_correct_axpy_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut a, mut x, mut y) = get_axpy_memory::<f32, Native>(&backend);

        if let Ok(_) = backend.axpy(&mut a, &mut x, &mut y) {
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[3f32, 6f32, 9f32], mem.as_slice::<f32>()) }
        }
        backend.axpy(&mut a, &mut x, &mut y).unwrap();
    }

    #[test]
    fn it_computes_correct_axpy_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut a, mut x, mut y) = get_axpy_memory::<f64, Native>(&backend);

        if let Ok(_) = backend.axpy(&mut a, &mut x, &mut y) {
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[3f64, 6f64, 9f64], mem.as_slice::<f64>()) }
        }
        backend.axpy(&mut a, &mut x, &mut y).unwrap();
    }

    #[test]
    fn it_computes_correct_copy_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut y) = get_copy_memory::<f32, Native>(&backend);

        if let Ok(_) = backend.copy(&mut x, &mut y) {
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[1f32, 2f32, 3f32], mem.as_slice::<f32>()) }
        }
        backend.copy(&mut x, &mut y).unwrap();
    }

    #[test]
    fn it_computes_correct_copy_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut y) = get_copy_memory::<f64, Native>(&backend);

        if let Ok(_) = backend.copy(&mut x, &mut y) {
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[1f64, 2f64, 3f64], mem.as_slice::<f64>()) }
        }
        backend.copy(&mut x, &mut y).unwrap();
    }

    #[test]
    fn it_computes_correct_dot_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut y, mut result) = get_dot_memory::<f32, Native>(&backend);

        if let Ok(_) = backend.dot(&mut x, &mut y, &mut result) {
            if let Some(mem) = result.get(backend.device()).unwrap().as_native() { assert_eq!(14f32, mem.as_slice::<f32>()[0]) }
        }
        backend.dot(&mut x, &mut y, &mut result).unwrap();
    }

    #[test]
    fn it_computes_correct_dot_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut y, mut result) = get_dot_memory::<f64, Native>(&backend);

        if let Ok(_) = backend.dot(&mut x, &mut y, &mut result) {
            if let Some(mem) = result.get(backend.device()).unwrap().as_native() { assert_eq!(14f64, mem.as_slice::<f64>()[0]) }
        }
        backend.dot(&mut x, &mut y, &mut result).unwrap();
    }

    // NRM2

    #[test]
    fn it_computes_correct_nrm2_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_nrm2_memory::<f32, Native>(&backend);

        if let Ok(_) = backend.nrm2(&mut x, &mut result) {
            if let Some(mem) = result.get(backend.device()).unwrap().as_native() { assert_eq!(3f32, mem.as_slice::<f32>()[0]) }
        }
        backend.nrm2(&mut x, &mut result).unwrap();
    }

    #[test]
    fn it_computes_correct_nrm2_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_nrm2_memory::<f64, Native>(&backend);

        if let Ok(_) = backend.nrm2(&mut x, &mut result) {
            if let Some(mem) = result.get(backend.device()).unwrap().as_native() { assert_eq!(3f64, mem.as_slice::<f64>()[0]) }
        }
        backend.nrm2(&mut x, &mut result).unwrap();
    }

    /// SCALE

    #[test]
    fn it_computes_correct_scale_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut y) = get_scale_memory::<f32, Native>(&backend);

        if let Ok(_) = backend.scale(&mut x, &mut y) {
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[2f32, 4f32, 6f32], mem.as_slice::<f32>()) }
        }
        backend.scale(&mut x, &mut y).unwrap();
    }

    #[test]
    fn it_computes_correct_scale_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut y) = get_scale_memory::<f64, Native>(&backend);

        if let Ok(_) = backend.scale(&mut x, &mut y) {
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[2f64, 4f64, 6f64], mem.as_slice::<f64>()) }
        }
        backend.scale(&mut x, &mut y).unwrap();
    }

    /// SWAP

    #[test]
    fn it_computes_correct_swap_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut y) = get_swap_memory::<f32, Native>(&backend);

        if let Ok(_) = backend.swap(&mut x, &mut y) {
            if let Some(mem) = x.get(backend.device()).unwrap().as_native() { assert_eq!(&[3f32, 2f32, 1f32], mem.as_slice::<f32>()) }
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[1f32, 2f32, 3f32], mem.as_slice::<f32>()) }
        }
        backend.swap(&mut x, &mut y).unwrap();
    }

    #[test]
    fn it_computes_correct_swap_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut y) = get_swap_memory::<f64, Native>(&backend);

        if let Ok(_) = backend.swap(&mut x, &mut y) {
            if let Some(mem) = x.get(backend.device()).unwrap().as_native() { assert_eq!(&[3f64, 2f64, 1f64], mem.as_slice::<f64>()) }
            if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[1f64, 2f64, 3f64], mem.as_slice::<f64>()) }
        }
        backend.swap(&mut x, &mut y).unwrap();
    }
}
