extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod blas_spec {

    use co::backend::{Backend, BackendConfig};
    use co::framework::IFramework;
    use co::frameworks::{OpenCL, Native};
    use co::libraries::blas::*;
    use co::memory::MemoryType;
    use co::shared_memory::SharedMemory;

    fn get_native_backend() -> Backend<Native> {
        let framework = Native::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Backend::new(backend_config).unwrap()
    }

    fn write_to_memory(mem: &mut MemoryType, data: &[f32]) {
        if let &mut MemoryType::Native(ref mut mem) = mem {
            let mut mem_buffer = mem.as_mut_slice::<f32>();
            for (index, datum) in data.iter().enumerate() {
                mem_buffer[index] = *datum;
            }
        }
    }

    #[test]
    fn it_computes_correct_dot_on_native() {
        let backend = get_native_backend();
        let mut x = &mut SharedMemory::<f32>::new(backend.device(), 3);
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[1f32, 2f32, 3f32]);

        let mut y = &mut SharedMemory::<f32>::new(backend.device(), 3);
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[1f32, 2f32, 3f32]);

        let mut result = &mut SharedMemory::<f32>::new(backend.device(), 1);
        backend.dot(x, y, result);

        match result.get(backend.device()).unwrap() {
            &MemoryType::Native(ref mem) => assert_eq!(14f32, mem.as_slice::<f32>()[0]),
            _ => assert_eq!(1, 2),
        }
    }
}
