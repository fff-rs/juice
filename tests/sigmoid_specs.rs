extern crate collenchyma_nn as co_nn;
extern crate collenchyma as co;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod sigmoid_spec_cuda{

    use co::backend::{Backend, BackendConfig};
    use co::framework::IFramework;
    use co::frameworks::{Cuda, Native};
    use co_nn::*;
    use co::memory::MemoryType;
    use co::tensor::SharedTensor;
    use co::plugin::numeric_helpers::{cast, Float};

    fn get_native_backend() -> Backend<Native> {
        let framework = Native::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Backend::new(backend_config).unwrap()
    }

    fn get_cuda_backend() -> Backend<Cuda> {
        let framework = Cuda::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Backend::new(backend_config).unwrap()
    }

    fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
        match mem {
            &mut MemoryType::Native(ref mut mem) => {
                let mut mem_buffer = mem.as_mut_slice::<T>();
                for (index, datum) in data.iter().enumerate() {
                    mem_buffer[index] = *datum;
                }
            },
            #[cfg(any(feature = "opencl", feature = "cuda"))]
            _ => {}
        }
    }

    fn get_memory<T: Float, B: IFramework + Clone, C: IFramework + Clone>(backend: &Backend<B>, native: &Backend<C>) -> (SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let mut x = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        x.add_device(native.device()).unwrap();
        x.sync(native.device()).unwrap();
        write_to_memory(x.get_mut(native.device()).unwrap(), &[val, val, val2]);
        x.sync(backend.device()).unwrap();

        let mut result = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        result.add_device(native.device()).unwrap();

        (x, result)
    }

    fn get_grad_memory<T: Float, B: IFramework + Clone, C: IFramework + Clone>(backend: &Backend<B>, native: &Backend<C>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let mut x = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        x.add_device(native.device()).unwrap();
        x.sync(native.device()).unwrap();
        write_to_memory(x.get_mut(native.device()).unwrap(), &[val, val, val2]);
        x.sync(backend.device()).unwrap();

        let mut x_diff = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        x_diff.add_device(native.device()).unwrap();
        x_diff.sync(native.device()).unwrap();
        write_to_memory(x_diff.get_mut(native.device()).unwrap(), &[val, val, val2]);
        x_diff.sync(backend.device()).unwrap();

        let mut result = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        result.add_device(native.device()).unwrap();
        result.sync(native.device()).unwrap();
        write_to_memory(result.get_mut(native.device()).unwrap(), &[val, val, val2]);
        result.sync(backend.device()).unwrap();

        let mut result_diff = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        result_diff.add_device(native.device()).unwrap();

        (x, x_diff, result, result_diff)
    }

    #[test]
    fn it_computes_correct_sigmoid_on_cuda_for_f32() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result) = get_memory::<f32, Cuda, Native>(&backend, &native);

        result.sync(native.device()).unwrap();
        result.sync(backend.device()).unwrap();
        result.sync(backend.device()).unwrap();
        result.sync(native.device()).unwrap();
        result.sync(backend.device()).unwrap();
        match backend.sigmoid(&mut x, &mut result) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_on_cuda_for_f64() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result) = get_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.sigmoid(&mut x, &mut result) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786300049f64, 0.7310585786300049f64, 0.8807970779778823f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_on_cuda_for_f32_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result) = get_memory::<f32, Cuda, Native>(&backend, &native);

        match backend.sigmoid_plain(&mut x, &mut result) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_on_cuda_for_f64_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result) = get_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.sigmoid_plain(&mut x, &mut result) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786300049f64, 0.7310585786300049f64, 0.8807970779778823f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_cuda_for_f32() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

        match backend.sigmoid_grad(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                result_diff.sync(native.device()).unwrap();
                if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0f32, 0f32, -4f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_cuda_for_f64() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.sigmoid_grad(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                result_diff.sync(native.device()).unwrap();
                if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0f64, 0f64, -4f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_cuda_for_f32_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

        match backend.sigmoid_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                result_diff.sync(native.device()).unwrap();
                if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0f32, 0f32, -4f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_cuda_for_f64_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.sigmoid_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                result_diff.sync(native.device()).unwrap();
                if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[0f64, 0f64, -4f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "native")]
mod sigmoid_spec_native {

    use co::backend::{Backend, BackendConfig};
    use co::framework::IFramework;
    use co::frameworks::Native;
    use co_nn::*;
    use co::memory::MemoryType;
    use co::tensor::SharedTensor;
    use co::plugin::numeric_helpers::{cast, Float};

    fn get_native_backend() -> Backend<Native> {
        let framework = Native::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Backend::new(backend_config).unwrap()
    }

    fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
        match mem {
            &mut MemoryType::Native(ref mut mem) => {
                let mut mem_buffer = mem.as_mut_slice::<T>();
                for (index, datum) in data.iter().enumerate() {
                    mem_buffer[index] = *datum;
                }
            },
            #[cfg(any(feature = "opencl", feature = "cuda"))]
            _ => {}
        }
    }

    fn get_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let mut x = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[val, val, val2]);

        let result = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();

        (x, result)
    }

    fn get_grad_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let mut x = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[val, val, val2]);

        let mut x_diff = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        write_to_memory(x_diff.get_mut(backend.device()).unwrap(), &[val, val, val2]);

        let mut result = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        write_to_memory(result.get_mut(backend.device()).unwrap(), &[val, val, val2]);

        let result_diff = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();

        (x, x_diff, result, result_diff)
    }

    #[test]
    fn it_computes_correct_sigmoid_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_memory::<f32, Native>(&backend);

        match backend.sigmoid(&mut x, &mut result) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], mem.as_slice::<f32>());
                } else {
                    println!("No result: {:?}", result); assert!(false);
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_memory::<f64, Native>(&backend);

        match backend.sigmoid(&mut x, &mut result) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786300049f64, 0.7310585786300049f64, 0.8807970779778823f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_on_native_for_f32_plain() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_memory::<f32, Native>(&backend);

        match backend.sigmoid_plain(&mut x, &mut result) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_on_native_for_f64_plain() {
        let backend = get_native_backend();
        let (mut x, mut result) = get_memory::<f64, Native>(&backend);

        match backend.sigmoid_plain(&mut x, &mut result) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0.7310585786300049f64, 0.7310585786300049f64, 0.8807970779778823f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f32, Native>(&backend);

        match backend.sigmoid_grad(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0f32, 0f32, -4f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f64, Native>(&backend);

        match backend.sigmoid_grad(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0f64, 0f64, -4f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_native_for_f32_plain() {
        let backend = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f32, Native>(&backend);

        match backend.sigmoid_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0f32, 0f32, -4f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_sigmoid_grad_on_native_for_f64_plain() {
        let backend = get_native_backend();
        let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f64, Native>(&backend);

        match backend.sigmoid_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff) {
            Ok(_) => {
                if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
                    assert_eq!(&[0f64, 0f64, -4f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }
}
