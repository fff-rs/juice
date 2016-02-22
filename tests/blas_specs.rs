extern crate collenchyma_blas as co_blas;
extern crate collenchyma as co;

#[cfg(test)]
mod blas_spec {
    use co::backend::Backend;
    use co::framework::IFramework;
    use co_blas::plugin::*;
    use co::memory::MemoryType;
    use co::tensor::SharedTensor;
    use co::plugin::numeric_helpers::{cast, Float};

    pub fn write_to_memory<T: ::std::marker::Copy>(mem: &mut MemoryType, data: &[T]) {
        match mem {
            &mut MemoryType::Native(ref mut mem) => {
                let mut mem_buffer = mem.as_mut_slice::<T>();
                for (index, datum) in data.iter().enumerate() {
                    mem_buffer[index] = *datum;
                }
            },
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => assert!(false)
        }
    }

    pub fn get_asum_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
        let mut x = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(-2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let result = SharedTensor::<T>::new(backend.device(), &()).unwrap();
        (x, result)
    }

    pub fn get_axpy_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let mut a = SharedTensor::<T>::new(backend.device(), &()).unwrap();
        write_to_memory(a.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(2).unwrap()]);

        let mut x = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);
        (a, x, y)
    }

    pub fn get_copy_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
        let mut x = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let y = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        (x, y)
    }

    pub fn get_dot_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let mut x = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let result = SharedTensor::<T>::new(backend.device(), &()).unwrap();
        (x, y, result)
    }

    pub fn get_nrm2_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
        let mut x = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(2).unwrap()]);

        let result = SharedTensor::<T>::new(backend.device(), &()).unwrap();
        (x, result)
    }

    pub fn get_scal_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
        let mut a = SharedTensor::<T>::new(backend.device(), &()).unwrap();
        write_to_memory(a.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(2).unwrap()]);

        let mut y = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        (a, y)
    }

    pub fn get_swap_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
        let mut x = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(x.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(3).unwrap()]);

        let mut y = SharedTensor::<T>::new(backend.device(), &vec![3]).unwrap();
        write_to_memory(y.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(3).unwrap(), cast::<i32, T>(2).unwrap(), cast::<i32, T>(1).unwrap()]);

        (x, y)
    }

    pub fn get_gemm_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let mut a = SharedTensor::<T>::new(backend.device(), &vec![3, 2]).unwrap();
        write_to_memory(a.get_mut(backend.device()).unwrap(),
            &[cast::<i32, T>(2).unwrap(), cast::<i32, T>(5).unwrap(),
              cast::<i32, T>(2).unwrap(), cast::<i32, T>(5).unwrap(),
              cast::<i32, T>(2).unwrap(), cast::<i32, T>(5).unwrap()]);

        let mut b = SharedTensor::<T>::new(backend.device(), &vec![2, 3]).unwrap();
        write_to_memory(b.get_mut(backend.device()).unwrap(),
            &[cast::<i32, T>(4).unwrap(), cast::<i32, T>(1).unwrap(), cast::<i32, T>(1).unwrap(),
              cast::<i32, T>(4).unwrap(), cast::<i32, T>(1).unwrap(), cast::<i32, T>(1).unwrap()]);

        let c = SharedTensor::<T>::new(backend.device(), &vec![3, 3]).unwrap();

        (a, b, c)
    }

    pub fn get_gemm_transpose_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>){
        let (a, b, _) = get_gemm_memory(backend);
        let c = SharedTensor::<T>::new(backend.device(), &vec![2, 2]).unwrap();

        (a, b, c)
    }

    pub fn get_scale_one_zero_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>){
        let mut alpha = SharedTensor::<T>::new(backend.device(), &vec![1]).unwrap();
        write_to_memory(alpha.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap()]);

        let mut beta = SharedTensor::<T>::new(backend.device(), &vec![1]).unwrap();
        write_to_memory(beta.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(0).unwrap()]);

        (alpha, beta)
    }

    #[cfg(feature = "native")]
    mod native {
        use co::backend::{Backend, BackendConfig};
        use co::framework::IFramework;
        use co::frameworks::Native;
        use co_blas::plugin::*;
        use co_blas::transpose::Transpose;
        use super::*;

        fn get_native_backend() -> Backend<Native> {
            let framework = Native::new();
            let hardwares = framework.hardwares().to_vec();
            let backend_config = BackendConfig::new(framework, &hardwares);
            Backend::new(backend_config).unwrap()
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

        /// SCAL

        #[test]
        fn it_computes_correct_scal_on_native_for_f32() {
            let backend = get_native_backend();
            let (mut x, mut y) = get_scal_memory::<f32, Native>(&backend);

            if let Ok(_) = backend.scal(&mut x, &mut y) {
                if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[2f32, 4f32, 6f32], mem.as_slice::<f32>()) }
            }
            backend.scal(&mut x, &mut y).unwrap();
        }

        #[test]
        fn it_computes_correct_scal_on_native_for_f64() {
            let backend = get_native_backend();
            let (mut x, mut y) = get_scal_memory::<f64, Native>(&backend);

            if let Ok(_) = backend.scal(&mut x, &mut y) {
                if let Some(mem) = y.get(backend.device()).unwrap().as_native() { assert_eq!(&[2f64, 4f64, 6f64], mem.as_slice::<f64>()) }
            }
            backend.scal(&mut x, &mut y).unwrap();
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

        /// GEMM

        #[test]
        fn it_computes_correct_gemm_on_native_for_f32() {
            let backend = get_native_backend();
            let (mut a, mut b, mut c) = get_gemm_memory::<f32, Native>(&backend);
            let (mut alpha, mut beta) = get_scale_one_zero_memory::<f32, Native>(&backend);

            if let Some(mem) = a.get(backend.device()).unwrap().as_native() { assert_eq!(&[2f32, 5f32, 2f32, 5f32, 2f32, 5f32], mem.as_slice::<f32>()) }
            if let Some(mem) = b.get(backend.device()).unwrap().as_native() { assert_eq!(&[4f32, 1f32, 1f32, 4f32, 1f32, 1f32], mem.as_slice::<f32>()) }
            if let Ok(_) = backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c) {
                if let Some(mem) = c.get(backend.device()).unwrap().as_native() { assert_eq!(&[28f32, 7f32, 7f32, 28f32, 7f32, 7f32, 28f32, 7f32, 7f32], mem.as_slice::<f32>()) }
            }
            backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c).unwrap();
        }

        #[test]
        fn it_computes_correct_gemm_on_native_for_f64() {
            let backend = get_native_backend();
            let (mut a, mut b, mut c) = get_gemm_memory::<f64, Native>(&backend);
            let (mut alpha, mut beta) = get_scale_one_zero_memory::<f64, Native>(&backend);

            if let Ok(_) = backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c) {
                if let Some(mem) = c.get(backend.device()).unwrap().as_native() { assert_eq!(&[28f64, 7f64, 7f64, 28f64, 7f64, 7f64, 28f64, 7f64, 7f64], mem.as_slice::<f64>()) }
            }
            backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c).unwrap();
        }

        #[test]
        fn it_computes_correct_gemm_transpose_on_native_for_f32() {
            let backend = get_native_backend();
            let (mut a, mut b, mut c) = get_gemm_transpose_memory::<f32, Native>(&backend);
            let (mut alpha, mut beta) = get_scale_one_zero_memory::<f32, Native>(&backend);

            if let Some(mem) = a.get(backend.device()).unwrap().as_native() { assert_eq!(&[2f32, 5f32, 2f32, 5f32, 2f32, 5f32], mem.as_slice::<f32>()) }
            if let Some(mem) = b.get(backend.device()).unwrap().as_native() { assert_eq!(&[4f32, 1f32, 1f32, 4f32, 1f32, 1f32], mem.as_slice::<f32>()) }
            if let Ok(_) = backend.gemm(&mut alpha, Transpose::Trans, &mut a, Transpose::Trans, &mut b, &mut beta, &mut c) {
                if let Some(mem) = c.get(backend.device()).unwrap().as_native() { assert_eq!(&[12f32, 12f32, 30f32, 30f32], mem.as_slice::<f32>()) }
            }
            backend.gemm(&mut alpha, Transpose::Trans, &mut a, Transpose::Trans, &mut b, &mut beta, &mut c).unwrap();
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use co::backend::{IBackend, Backend, BackendConfig};
        use co::framework::IFramework;
        use co::frameworks::Native;
        use co::frameworks::Cuda;
        use co_blas::plugin::*;
        use co_blas::transpose::Transpose;
        use super::*;

        fn get_native_backend() -> Backend<Native> {
            let framework = Native::new();
            let hardwares = framework.hardwares().to_vec();
            let backend_config = BackendConfig::new(framework, &hardwares);
            Backend::new(backend_config).unwrap()
        }

        fn get_cuda_backend() -> Backend<Cuda> {
            let framework = Cuda::new();
            let hardwares = framework.hardwares().to_vec();
            let backend_config = BackendConfig::new(framework, &hardwares);
            Backend::new(backend_config).unwrap()
        }

        #[test]
        fn it_computes_correct_asum_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut x, mut result) = get_asum_memory::<f32, Native>(&native);

            if let Ok(_) = backend.asum(&mut x, &mut result) {
                backend.synchronize().unwrap();
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() { assert_eq!(&[6f32], mem.as_slice::<f32>()) }
            }
            backend.asum(&mut x, &mut result).unwrap();
        }

        #[test]
        fn it_computes_correct_axpy_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut a, mut x, mut y) = get_axpy_memory::<f32, Native>(&native);

            if let Ok(_) = backend.axpy(&mut a, &mut x, &mut y) {
                backend.synchronize().unwrap();
                y.sync(native.device()).unwrap();
                if let Some(mem) = y.get(native.device()).unwrap().as_native() { assert_eq!(&[3f32, 6f32, 9f32], mem.as_slice::<f32>()) }
            }
            backend.axpy(&mut a, &mut x, &mut y).unwrap();
        }

        #[test]
        fn it_computes_correct_copy_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut x, mut y) = get_copy_memory::<f32, Native>(&native);

            if let Ok(_) = backend.copy(&mut x, &mut y) {
                backend.synchronize().unwrap();
                y.sync(native.device()).unwrap();
                if let Some(mem) = y.get(native.device()).unwrap().as_native() { assert_eq!(&[1f32, 2f32, 3f32], mem.as_slice::<f32>()) }
            }
            backend.copy(&mut x, &mut y).unwrap();
        }

        #[test]
        // #[ignore]
        fn it_computes_correct_dot_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut x, mut y, mut result) = get_dot_memory::<f32, Native>(&native);
            backend.synchronize().unwrap();

            if let Ok(_) = backend.dot(&mut x, &mut y, &mut result) {
                println!("DOT");
                backend.synchronize().unwrap();
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() { println!("{:?}", mem.as_slice::<f32>()[0]); assert_eq!(14f32, mem.as_slice::<f32>()[0]) }
            }
            backend.dot(&mut x, &mut y, &mut result).unwrap();
        }

        #[test]
        // #[ignore]
        fn it_computes_correct_nrm2_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut x, mut result) = get_nrm2_memory::<f32, Native>(&native);

            if let Ok(_) = backend.nrm2(&mut x, &mut result) {
                backend.synchronize().unwrap();
                result.sync(native.device()).unwrap();
                if let Some(mem) = result.get(native.device()).unwrap().as_native() { assert_eq!(3f32, mem.as_slice::<f32>()[0]) }
            }
            backend.nrm2(&mut x, &mut result).unwrap();
        }

        #[test]
        fn it_computes_correct_scal_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut a, mut x) = get_scal_memory::<f32, Native>(&native);

            if let Ok(_) = backend.scal(&mut a, &mut x) {
                backend.synchronize().unwrap();
                x.sync(native.device()).unwrap();
                if let Some(mem) = x.get(native.device()).unwrap().as_native() { assert_eq!(&[2f32, 4f32, 6f32], mem.as_slice::<f32>()) }
            }
            backend.scal(&mut a, &mut x).unwrap();
        }

        #[test]
        fn it_computes_correct_swap_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut x, mut y) = get_swap_memory::<f32, Native>(&native);

            if let Ok(_) = backend.swap(&mut x, &mut y) {
                backend.synchronize().unwrap();
                x.sync(native.device()).unwrap();
                y.sync(native.device()).unwrap();
                if let Some(mem) = x.get(native.device()).unwrap().as_native() { assert_eq!(&[3f32, 2f32, 1f32], mem.as_slice::<f32>()) }
                if let Some(mem) = y.get(native.device()).unwrap().as_native() { assert_eq!(&[1f32, 2f32, 3f32], mem.as_slice::<f32>()) }
            }
            backend.swap(&mut x, &mut y).unwrap();
        }

        #[test]
        fn it_computes_correct_gemm_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut a, mut b, mut c) = get_gemm_memory::<f32, Native>(&native);
            let (mut alpha, mut beta) = get_scale_one_zero_memory::<f32, Native>(&native);

            if let Some(mem) = a.get(native.device()).unwrap().as_native() { assert_eq!(&[2f32, 5f32, 2f32, 5f32, 2f32, 5f32], mem.as_slice::<f32>()) }
            if let Some(mem) = b.get(native.device()).unwrap().as_native() { assert_eq!(&[4f32, 1f32, 1f32, 4f32, 1f32, 1f32], mem.as_slice::<f32>()) }
            if let Ok(_) = backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c) {
                backend.synchronize().unwrap();
                c.sync(native.device()).unwrap();
                if let Some(mem) = c.get(native.device()).unwrap().as_native() { assert_eq!(&[28f32, 7f32, 7f32, 28f32, 7f32, 7f32, 28f32, 7f32, 7f32], mem.as_slice::<f32>()) }
            }
            backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c).unwrap();
        }

        #[test]
        fn it_computes_correct_transpose_gemm_on_cuda_for_f32() {
            let native = get_native_backend();
            let backend = get_cuda_backend();
            let (mut a, mut b, mut c) = get_gemm_transpose_memory::<f32, Native>(&native);
            let (mut alpha, mut beta) = get_scale_one_zero_memory::<f32, Native>(&native);

            if let Some(mem) = a.get(native.device()).unwrap().as_native() { assert_eq!(&[2f32, 5f32, 2f32, 5f32, 2f32, 5f32], mem.as_slice::<f32>()) }
            if let Some(mem) = b.get(native.device()).unwrap().as_native() { assert_eq!(&[4f32, 1f32, 1f32, 4f32, 1f32, 1f32], mem.as_slice::<f32>()) }
            if let Ok(_) = backend.gemm(&mut alpha, Transpose::Trans, &mut a, Transpose::Trans, &mut b, &mut beta, &mut c) {
                backend.synchronize().unwrap();
                c.sync(native.device()).unwrap();
                if let Some(mem) = c.get(native.device()).unwrap().as_native() { assert_eq!(&[12f32, 12f32, 30f32, 30f32], &mem.as_slice::<f32>()) }
            }
            backend.gemm(&mut alpha, Transpose::Trans, &mut a, Transpose::Trans, &mut b, &mut beta, &mut c).unwrap();
        }
    }
}
