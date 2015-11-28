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
    use co::shared_memory::SharedMemory;
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

    fn get_axpy_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T>, SharedMemory<T>, SharedMemory<T>){
        let mut a = SharedMemory::<T>::new(backend.device(), 1);
        write_to_memory(a.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(2).unwrap()]);

        let mut x = SharedMemory::<T>::new(backend.device(), 3);
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedMemory::<T>::new(backend.device(), 3);
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);
        (a, x, y)
    }

    fn get_dot_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T>, SharedMemory<T>, SharedMemory<T>){
        let mut x = SharedMemory::<T>::new(backend.device(), 3);
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedMemory::<T>::new(backend.device(), 3);
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let result = SharedMemory::<T>::new(backend.device(), 1);
        (x, y, result)
    }

    fn get_scale_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedMemory<T>, SharedMemory<T>){
        let mut x = SharedMemory::<T>::new(backend.device(), 1);
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(2).unwrap()]);

        let mut y = SharedMemory::<T>::new(backend.device(), 3);
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        (x, y)
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

}
