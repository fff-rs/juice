extern crate collenchyma_nn as co_nn;
extern crate collenchyma as co;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod convolution_spec_cuda {

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

    fn get_memory<T: Float, B: IFramework + Clone, C: IFramework + Clone>(backend: &Backend<B>, native: &Backend<C>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let batch = 4;
        let w1 = 9;
        let h1 = 9;
        let d1 = 3;
        let k = 6;
        let f = 3;
        let w2 = (w1 - f + 0) / 1;
        let h2 = (h1 - f + 0) / 1;
        let mut x = SharedTensor::<T>::new(backend.device(), &(batch, d1, h1, w1)).unwrap();
        let mut payload: &mut [T] = &mut ::std::iter::repeat(val).take(x.capacity()).collect::<Vec<T>>();
        payload[0] = val2;
        x.add_device(native.device()).unwrap();
        x.sync(native.device()).unwrap();
        write_to_memory(x.get_mut(native.device()).unwrap(), payload);
        x.sync(backend.device()).unwrap();

        let mut filter = SharedTensor::<T>::new(backend.device(), &(k, d1, f, f)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val).take(filter.capacity()).collect::<Vec<T>>();
        filter.add_device(native.device()).unwrap();
        filter.sync(native.device()).unwrap();
        write_to_memory(filter.get_mut(native.device()).unwrap(), payload);
        filter.sync(backend.device()).unwrap();

        let mut result = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val2).take(result.capacity()).collect::<Vec<T>>();
        result.add_device(native.device()).unwrap();
        result.sync(native.device()).unwrap();
        write_to_memory(result.get_mut(native.device()).unwrap(), payload);
        result.sync(backend.device()).unwrap();

        (x, result, filter)
    }

    #[allow(dead_code)]
    fn get_grad_memory<T: Float, B: IFramework + Clone, C: IFramework + Clone>(backend: &Backend<B>, native: &Backend<C>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let batch = 4;
        let w1 = 9;
        let h1 = 9;
        let d1 = 3;
        let k = 6;
        let f = 3;
        let w2 = (w1 - f + 0) / 1;
        let h2 = (h1 - f + 0) / 1;

        let mut x = SharedTensor::<T>::new(backend.device(), &(batch, d1, h1, w1)).unwrap();
        let mut payload: &mut [T] = &mut ::std::iter::repeat(val).take(x.capacity()).collect::<Vec<T>>();
        payload[0] = val2;
        x.add_device(native.device()).unwrap();
        x.sync(native.device()).unwrap();
        write_to_memory(x.get_mut(native.device()).unwrap(), payload);
        x.sync(backend.device()).unwrap();

        let mut x_diff = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();
        let mut payload: &mut [T] = &mut ::std::iter::repeat(val).take(x_diff.capacity()).collect::<Vec<T>>();
        payload[0] = val2;
        x_diff.add_device(native.device()).unwrap();
        x_diff.sync(native.device()).unwrap();
        write_to_memory(x_diff.get_mut(native.device()).unwrap(), payload);
        x_diff.sync(backend.device()).unwrap();

        let mut filter = SharedTensor::<T>::new(backend.device(), &(k, d1, f, f)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val).take(filter.capacity()).collect::<Vec<T>>();
        filter.add_device(native.device()).unwrap();
        filter.sync(native.device()).unwrap();
        write_to_memory(filter.get_mut(native.device()).unwrap(), payload);
        filter.sync(backend.device()).unwrap();

        let mut result = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val).take(result.capacity()).collect::<Vec<T>>();
        result.add_device(native.device()).unwrap();
        result.sync(native.device()).unwrap();
        write_to_memory(result.get_mut(native.device()).unwrap(), payload);
        result.sync(backend.device()).unwrap();

        let mut result_diff = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();
        result_diff.add_device(native.device()).unwrap();

        (x, x_diff, result, result_diff, filter)
    }

    #[test]
    fn it_computes_correct_convolution_on_cuda_for_f32() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f32, Cuda, Native>(&backend, &native);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution(&mut x, &mut result, &conf) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    let mut payload: &mut [f32] = &mut ::std::iter::repeat(27f32).take(result.capacity()).collect::<Vec<f32>>();
                    payload[0] = 28f32;
                    assert_eq!(payload, mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_convolution_on_cuda_for_f64() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f64, Cuda, Native>(&backend, &native);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution(&mut x, &mut result, &conf) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    let mut payload: &mut [f64] = &mut ::std::iter::repeat(27f64).take(result.capacity()).collect::<Vec<f64>>();
                    payload[0] = 28f64;
                    assert_eq!(payload, mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_convolution_on_cuda_for_f32_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f32, Cuda, Native>(&backend, &native);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution_plain(&mut x, &mut result, &conf) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    let mut payload: &mut [f32] = &mut ::std::iter::repeat(27f32).take(result.capacity()).collect::<Vec<f32>>();
                    payload[0] = 28f32;
                    assert_eq!(payload, mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_computes_correct_convolution_on_cuda_for_f64_plain() {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f64, Cuda, Native>(&backend, &native);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution_plain(&mut x, &mut result, &conf) {
            Ok(_) => {
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() {
                    let mut payload: &mut [f64] = &mut ::std::iter::repeat(27f64).take(result.capacity()).collect::<Vec<f64>>();
                    payload[0] = 28f64;
                    assert_eq!(payload, mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    /*
    #[test]
    fn it_computes_correct_convolution_grad_on_cuda_for_f32() {
    let backend = get_cuda_backend();
    let native = get_native_backend();
    let (mut x, mut x_diff, mut result, mut result_diff, mut filter) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

    let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
    match backend.convolution_grad(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
    Ok(_) => {
    result_diff.sync(native.device()).unwrap();
    if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
    assert_eq!(&[0f32, 0f32, -6f32], mem.as_slice::<f32>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}

#[test]
fn it_computes_correct_convolution_grad_on_cuda_for_f64() {
let backend = get_cuda_backend();
let native = get_native_backend();
let (mut x, mut x_diff, mut result, mut result_diff, filter, conv) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

let conf = backend.new_convolution_config(&x, &result, &filter, &conv).unwrap();
match backend.convolution_grad(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
Ok(_) => {
result_diff.sync(native.device()).unwrap();
if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
assert_eq!(&[0f64, 0f64, -6f64], mem.as_slice::<f64>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}

#[test]
fn it_computes_correct_convolution_grad_on_cuda_for_f32_plain() {
let backend = get_cuda_backend();
let native = get_native_backend();
let (mut x, mut x_diff, mut result, mut result_diff, filter, conv) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

let conf = backend.new_convolution_config(&x, &result, &filter, &conv).unwrap();
match backend.convolution_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
Ok(_) => {
result_diff.sync(native.device()).unwrap();
if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
assert_eq!(&[0f32, 0f32, -6f32], mem.as_slice::<f32>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}

#[test]
fn it_computes_correct_convolution_grad_on_cuda_for_f64_plain() {
let backend = get_cuda_backend();
let native = get_native_backend();
let (mut x, mut x_diff, mut result, mut result_diff, filter, conv) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

let conf = backend.new_convolution_config(&x, &result, &filter, &conv).unwrap();
match backend.convolution_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
Ok(_) => {
result_diff.sync(native.device()).unwrap();
if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
assert_eq!(&[0f64, 0f64, -6f64], mem.as_slice::<f64>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}
*/
}

#[cfg(test)]
#[cfg(feature = "native")]
mod convolution_spec_native{

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

    fn get_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let batch = 4;
        let w1 = 9;
        let h1 = 9;
        let d1 = 3;
        let k = 6;
        let f = 3;
        let w2 = (w1 - f + 0) / 1;
        let h2 = (h1 - f + 0) / 1;
        let mut x = SharedTensor::<T>::new(backend.device(), &(batch, d1, h1, w1)).unwrap();
        let mut payload: &mut [T] = &mut ::std::iter::repeat(val).take(x.capacity()).collect::<Vec<T>>();
        payload[0] = val2;
        write_to_memory(x.get_mut(backend.device()).unwrap(), payload);

        let mut filter = SharedTensor::<T>::new(backend.device(), &(k, d1, f, f)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val).take(filter.capacity()).collect::<Vec<T>>();
        write_to_memory(filter.get_mut(backend.device()).unwrap(), payload);

        let mut result = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val2).take(result.capacity()).collect::<Vec<T>>();
        write_to_memory(result.get_mut(backend.device()).unwrap(), payload);

        (x, result, filter)
    }

    #[allow(dead_code)]
    fn get_grad_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let val = cast::<f64, T>(1f64).unwrap();
        let val2 = cast::<f64, T>(2f64).unwrap();
        let batch = 4;
        let w1 = 9;
        let h1 = 9;
        let d1 = 3;
        let k = 6;
        let f = 3;
        let w2 = (w1 - f + 0) / 1;
        let h2 = (h1 - f + 0) / 1;

        let mut x = SharedTensor::<T>::new(backend.device(), &(batch, d1, h1, w1)).unwrap();
        let mut payload: &mut [T] = &mut ::std::iter::repeat(val).take(x.capacity()).collect::<Vec<T>>();
        payload[0] = val2;
        write_to_memory(x.get_mut(backend.device()).unwrap(), payload);

        let mut x_diff = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();
        let mut payload: &mut [T] = &mut ::std::iter::repeat(val).take(x_diff.capacity()).collect::<Vec<T>>();
        payload[0] = val2;
        write_to_memory(x_diff.get_mut(backend.device()).unwrap(), payload);

        let mut filter = SharedTensor::<T>::new(backend.device(), &(k, d1, f, f)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val).take(filter.capacity()).collect::<Vec<T>>();
        write_to_memory(filter.get_mut(backend.device()).unwrap(), payload);

        let mut result = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();
        let payload: &[T] = &::std::iter::repeat(val).take(result.capacity()).collect::<Vec<T>>();
        write_to_memory(result.get_mut(backend.device()).unwrap(), payload);

        let result_diff = SharedTensor::<T>::new(backend.device(), &(batch, k, h2, w2)).unwrap();

        (x, x_diff, result, result_diff, filter)
    }

    #[test]
    #[ignore]
    fn it_computes_correct_convolution_on_native_for_f32() {
        let backend = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f32, Native>(&backend);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution(&mut x, &mut result, &conf) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    let mut payload: &mut [f32] = &mut ::std::iter::repeat(27f32).take(result.capacity()).collect::<Vec<f32>>();
                    payload[0] = 28f32;
                    assert_eq!(payload, mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    #[ignore]
    fn it_computes_correct_convolution_on_native_for_f64() {
        let backend = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f64, Native>(&backend);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution(&mut x, &mut result, &conf) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    let mut payload: &mut [f64] = &mut ::std::iter::repeat(27f64).take(result.capacity()).collect::<Vec<f64>>();
                    payload[0] = 28f64;
                    assert_eq!(payload, mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    #[ignore]
    fn it_computes_correct_convolution_on_native_for_f32_plain() {
        let backend = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f32, Native>(&backend);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution_plain(&mut x, &mut result, &conf) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    let mut payload: &mut [f32] = &mut ::std::iter::repeat(27f32).take(result.capacity()).collect::<Vec<f32>>();
                    payload[0] = 28f32;
                    assert_eq!(payload, mem.as_slice::<f32>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    #[ignore]
    fn it_computes_correct_convolution_on_native_for_f64_plain() {
        let backend = get_native_backend();
        let (mut x, mut result, mut filter) = get_memory::<f64, Native>(&backend);

        let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
        match backend.convolution_plain(&mut x, &mut result, &conf) {
            Ok(_) => {
                if let Some(mem) = result.get(backend.device()).unwrap().as_native() {
                    let mut payload: &mut [f64] = &mut ::std::iter::repeat(27f64).take(result.capacity()).collect::<Vec<f64>>();
                    payload[0] = 28f64;
                    assert_eq!(payload, mem.as_slice::<f64>());
                }
            },
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }
}
