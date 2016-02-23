extern crate collenchyma_nn as co_nn;
extern crate collenchyma as co;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod relu_pointwise_spec_cuda{

    use co::prelude::*;
    use co_nn::*;
    use co::plugin::numeric_helpers::{cast, Float};

    fn get_native_backend() -> Backend<Native> {
        Backend::<Native>::default().unwrap()
    }

    fn get_cuda_backend() -> Backend<Cuda> {
        Backend::<Cuda>::default().unwrap()
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

    fn get_memory<T: Float, B: IFramework + Clone, C: IFramework + Clone>(backend: &Backend<B>, native: &Backend<C>) -> SharedTensor<T>{
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let mut x = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
        x.add_device(native.device()).unwrap();
        x.sync(native.device()).unwrap();
        write_to_memory(x.get_mut(native.device()).unwrap(), &[val, val, val2]);
        x.sync(backend.device()).unwrap();

        x
    }

    fn get_grad_memory<T: Float, B: IFramework + Clone, C: IFramework + Clone>(backend: &Backend<B>, native: &Backend<C>) -> (SharedTensor<T>, SharedTensor<T>){
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

        (x, x_diff)
    }

    #[test]
    fn it_computes_correct_relu_on_cuda_for_f32() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let mut x = get_memory::<f32, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise(&mut x) {
            Ok(_) => {
                x.sync(native.device()).unwrap();
                if let Some(mem) = x.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f32, 1f32, 2f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_relu_on_cuda_for_f64() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let mut x = get_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise(&mut x) {
            Ok(_) => {
                x.sync(native.device()).unwrap();
                if let Some(mem) = x.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f64, 1f64, 2f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_relu_on_cuda_for_f32_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let mut x = get_memory::<f32, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise_plain(&mut x) {
            Ok(_) => {
                x.sync(native.device()).unwrap();
                if let Some(mem) = x.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f32, 1f32, 2f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_relu_on_cuda_for_f64_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let mut x = get_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise_plain(&mut x) {
            Ok(_) => {
                x.sync(native.device()).unwrap();
                if let Some(mem) = x.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f64, 1f64, 2f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_relu_grad_on_cuda_for_f32() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise_grad(&mut x, &mut x_diff) {
            Ok(_) => {
                x_diff.sync(native.device()).unwrap();
                if let Some(mem) = x_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f32, 1f32, 2f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_relu_grad_on_cuda_for_f64() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise_grad(&mut x, &mut x_diff) {
            Ok(_) => {
                x_diff.sync(native.device()).unwrap();
                if let Some(mem) = x_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f64, 1f64, 2f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_relu_grad_on_cuda_for_f32_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise_grad_plain(&mut x, &mut x_diff) {
            Ok(_) => {
                x_diff.sync(native.device()).unwrap();
                if let Some(mem) = x_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f32, 1f32, 2f32], mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_relu_grad_on_cuda_for_f64_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut x_diff) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

        match backend.relu_pointwise_grad_plain(&mut x, &mut x_diff) {
            Ok(_) => {
                x_diff.sync(native.device()).unwrap();
                if let Some(mem) = x_diff.get(native.device()).unwrap().as_native() {
                    assert_eq!(&[1f64, 1f64, 2f64], mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "native")]
mod relu_pointwise_spec_native {

    // use co::prelude::*;
    // use co_nn::*;
    // use co::plugin::numeric_helpers::{cast, Float};
    //
    // fn get_native_backend() -> Backend<Native> {
    //     Backend::<Native>::default().unwrap()
    // }
    //
    // fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
    //     match mem {
    //         &mut MemoryType::Native(ref mut mem) => {
    //             let mut mem_buffer = mem.as_mut_slice::<T>();
    //             for (index, datum) in data.iter().enumerate() {
    //                 mem_buffer[index] = *datum;
    //             }
    //         },
    //         #[cfg(any(feature = "opencl", feature = "cuda"))]
    //         _ => {}
    //     }
    // }
    //
    // fn get_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
    //     let val = cast::<f64, T>(1f64).unwrap();
    //     let val2 = cast::<f64, T>(2f64).unwrap();
    //     let mut x = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
    //     write_to_memory(x.get_mut(backend.device()).unwrap(), &[val, val, val2]);
    //
    //     let result = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
    //
    //     (x, result)
    // }
    //
    // fn get_grad_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
    //     let val = cast::<f64, T>(1f64).unwrap();
    //     let val2 = cast::<f64, T>(2f64).unwrap();
    //     let mut x = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
    //     write_to_memory(x.get_mut(backend.device()).unwrap(), &[val, val, val2]);
    //
    //     let mut x_diff = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
    //     write_to_memory(x_diff.get_mut(backend.device()).unwrap(), &[val, val, val2]);
    //
    //     let mut result = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
    //     write_to_memory(result.get_mut(backend.device()).unwrap(), &[val, val, val2]);
    //
    //     let result_diff = SharedTensor::<T>::new(backend.device(), &(1, 1, 3)).unwrap();
    //
    //     (x, x_diff, result, result_diff)
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_on_native_for_f32() {
    //     let backend = get_native_backend();
    //     let (mut x, mut result) = get_memory::<f32, Native>(&backend);
    //
    //     match backend.relu(&mut x, &mut result) {
    //         Ok(_) => {
    //             if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], mem.as_slice::<f32>());
    //             } else {
    //                 println!("No result: {:?}", result); assert!(false);
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_on_native_for_f64() {
    //     let backend = get_native_backend();
    //     let (mut x, mut result) = get_memory::<f64, Native>(&backend);
    //
    //     match backend.relu(&mut x, &mut result) {
    //         Ok(_) => {
    //             if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0.7310585786300049f64, 0.7310585786300049f64, 0.8807970779778823f64], mem.as_slice::<f64>());
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_on_native_for_f32_plain() {
    //     let backend = get_native_backend();
    //     let (mut x, mut result) = get_memory::<f32, Native>(&backend);
    //
    //     match backend.relu_plain(&mut x, &mut result) {
    //         Ok(_) => {
    //             if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0.7310585786f32, 0.7310586f32, 0.880797f32], mem.as_slice::<f32>());
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_on_native_for_f64_plain() {
    //     let backend = get_native_backend();
    //     let (mut x, mut result) = get_memory::<f64, Native>(&backend);
    //
    //     match backend.relu_plain(&mut x, &mut result) {
    //         Ok(_) => {
    //             if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0.7310585786300049f64, 0.7310585786300049f64, 0.8807970779778823f64], mem.as_slice::<f64>());
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_grad_on_native_for_f32() {
    //     let backend = get_native_backend();
    //     let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f32, Native>(&backend);
    //
    //     match backend.relu_grad(&mut x, &mut x_diff, &mut result, &mut result_diff) {
    //         Ok(_) => {
    //             if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0f32, 0f32, -4f32], mem.as_slice::<f32>());
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_grad_on_native_for_f64() {
    //     let backend = get_native_backend();
    //     let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f64, Native>(&backend);
    //
    //     match backend.relu_grad(&mut x, &mut x_diff, &mut result, &mut result_diff) {
    //         Ok(_) => {
    //             if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0f64, 0f64, -4f64], mem.as_slice::<f64>());
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_grad_on_native_for_f32_plain() {
    //     let backend = get_native_backend();
    //     let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f32, Native>(&backend);
    //
    //     match backend.relu_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff) {
    //         Ok(_) => {
    //             if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0f32, 0f32, -4f32], mem.as_slice::<f32>());
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
    //
    // #[test]
    // fn it_computes_correct_relu_grad_on_native_for_f64_plain() {
    //     let backend = get_native_backend();
    //     let (mut x, mut x_diff, mut result, mut result_diff) = get_grad_memory::<f64, Native>(&backend);
    //
    //     match backend.relu_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff) {
    //         Ok(_) => {
    //             if let Some(mem) = result_diff.get(backend.device()).unwrap().as_native() {
    //                 assert_eq!(&[0f64, 0f64, -4f64], mem.as_slice::<f64>());
    //             }
    //         },
    //         Err(err) => { println!("{:?}", err); assert!(false) }
    //     }
    // }
}
