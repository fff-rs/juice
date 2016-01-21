use ::{API, Error};
use super::Context;
use ffi::*;

impl API {
    // TODO: cublasIsamax_v2 x 4
    // TODO: cublasIsamin_v2 x 4

    // TODO: cublasSasum_v2 x 4

    /// Compute the sum of magnitudes of the provided vector elements.
    ///
    /// `x`: pointer to input vector.
    /// `result`: pointer to output scalar.
    /// `n`: number of elements to compute sum over (should not be greater than `x`).
    /// `stride`: offset from one input element to the next. Defaults to `1`.
    pub fn asum(context: &Context, x: *mut f32, result: *mut f32, n: i32, stride: Option<i32>) -> Result<(), Error> {
        let stride_x = stride.unwrap_or(1);
        unsafe { Self::ffi_sasum(*context.id_c(), n, x, stride_x, result) }
    }

    unsafe fn ffi_sasum(handle: cublasHandle_t, n: i32, x: *mut f32, incx: i32, result: *mut f32) -> Result<(), Error> {
        match cublasSasum_v2(handle, n, x, incx, result) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => Err(Error::AllocFailed),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to calculate sum of x.")),
        }
    }

    // TODO: cublasSaxpy_v2 x 4

    /// Computes a vector-scalar product and adds the result to a vector.
    ///
    /// `alpha`: pointer to input scalar.
    /// `x`: pointer to input vector.
    /// `y`: pointer to output vector.
    /// `n`: number of elements to use for operation (should not be greater than number of elements in `x` or `y`).
    /// `stride_x`: offset from one element in x to the next. Defaults to `1`.
    /// `stride_y`: offset from one element in y to the next. Defaults to `1`.
    pub fn axpy(context: &Context, alpha: *mut f32, x: *mut f32, y: *mut f32, n: i32, stride_x: Option<i32>, stride_y: Option<i32>) -> Result<(), Error> {
        let stride_x = stride_x.unwrap_or(1);
        let stride_y = stride_y.unwrap_or(1);
        unsafe { Self::ffi_saxpy(*context.id_c(), n, alpha, x, stride_x, y, stride_y) }
    }

    unsafe fn ffi_saxpy(handle: cublasHandle_t, n: i32, alpha: *mut f32, x: *mut f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        match cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to calculate axpy (alpha * x + y).")),
        }
    }

    // TODO: cublasScopy_v2 x 4

    /// Copies a vector into another vector.
    ///
    /// `x`: pointer to input vector.
    /// `y`: pointer to output vector.
    /// `n`: number of elements to use for operation (should not be greater than number of elements in `x` or `y`).
    /// `stride_x`: offset from one element in x to the next. Defaults to `1`.
    /// `stride_y`: offset from one element in y to the next. Defaults to `1`.
    pub fn copy(context: &Context, x: *mut f32, y: *mut f32, n: i32, stride_x: Option<i32>, stride_y: Option<i32>) -> Result<(), Error> {
        let stride_x = stride_x.unwrap_or(1);
        let stride_y = stride_y.unwrap_or(1);
        unsafe { Self::ffi_scopy(*context.id_c(), n, x, stride_x, y, stride_y) }
    }

    unsafe fn ffi_scopy(handle: cublasHandle_t, n: i32, x: *mut f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        match cublasScopy_v2(handle, n, x, incx, y, incy) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to calculate copy from x to y.")),
        }
    }

    // TODO: cublasSdot_v2 x 6

    /// TODO: DOC
    pub fn dot(context: &Context, x: *mut f32, y: *mut f32, result: *mut f32, n: i32, stride_x: Option<i32>, stride_y: Option<i32>) -> Result<(), Error> {
        let stride_x = stride_x.unwrap_or(1);
        let stride_y = stride_y.unwrap_or(1);
        unsafe { Self::ffi_sdot(*context.id_c(), n, x, stride_x, y, stride_y, result) }
    }

    unsafe fn ffi_sdot(handle: cublasHandle_t, n: i32, x: *mut f32, incx: i32, y: *mut f32, incy: i32, result: *mut f32) -> Result<(), Error> {
        match cublasSdot_v2(handle, n, x, incx, y, incy, result) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to calculate dot product of x and y.")),
        }
    }

    // TODO: cublasSnrm2_v2 x 4

    /// TODO: DOC
    pub fn nrm2(context: &Context, x: *mut f32, result: *mut f32, n: i32, stride_x: Option<i32>) -> Result<(), Error> {
        let stride_x = stride_x.unwrap_or(1);
        unsafe { Self::ffi_snrm2(*context.id_c(), n, x, stride_x, result) }
    }

    unsafe fn ffi_snrm2(handle: cublasHandle_t, n: i32, x: *mut f32, incx: i32, result: *mut f32) -> Result<(), Error> {
        match cublasSnrm2_v2(handle, n, x, incx, result) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to calculate the euclidian norm of x.")),
        }
    }

    // TODO: cublasSrot_v2 x 6
    // TODO: cublasSrotg_v2 x 4
    // TODO: cublasSrotm_v2 x 2
    // TODO: cublasSrotmg_v2 x 2

    // TODO: cublasSscal_v2 x 6

    /// TODO: DOC
    pub fn scal(context: &Context, alpha: *mut f32, x: *mut f32, n: i32, stride_x: Option<i32>) -> Result<(), Error> {
        let stride_x = stride_x.unwrap_or(1);
        unsafe { Self::ffi_sscal(*context.id_c(), n, alpha, x, stride_x) }
    }

    unsafe fn ffi_sscal(handle: cublasHandle_t, n: i32, alpha: *mut f32, x: *mut f32, incx: i32) -> Result<(), Error> {
        match cublasSscal_v2(handle, n, alpha, x, incx) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to scale the vector x.")),
        }
    }

    // TODO: cublasSswap_v2 x 4

    /// TODO: DOC
    pub fn swap(context: &Context, x: *mut f32, y: *mut f32, n: i32, stride_x: Option<i32>, stride_y: Option<i32>) -> Result<(), Error> {
        let stride_x = stride_x.unwrap_or(1);
        let stride_y = stride_y.unwrap_or(1);
        unsafe { Self::ffi_sswap(*context.id_c(), n, x, stride_x, y, stride_y) }
    }

    unsafe fn ffi_sswap(handle: cublasHandle_t, n: i32, x: *mut f32, incx: i32, y: *mut f32, incy: i32) -> Result<(), Error> {
        match cublasSswap_v2(handle, n, x, incx, y, incy) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to swap vector x and y.")),
        }
    }
}

#[cfg(test)]
mod test {
    use ::API;
    use ::api::context::Context;
    use ::api::enums::PointerMode;
    use co::backend::{Backend, IBackend, BackendConfig};
    use co::framework::IFramework;
    use co::frameworks::{Cuda, Native};
    use co::tensor::SharedTensor;
    use co::memory::MemoryType;

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
        if let &mut MemoryType::Native(ref mut mem) = mem {
            let mut mem_buffer = mem.as_mut_slice::<T>();
            for (index, datum) in data.iter().enumerate() {
                mem_buffer[index] = *datum;
            }
        }
    }

    fn filled_tensor<B: IBackend, T: Copy>(backend: &B, n: usize, val: T) -> SharedTensor<T> {
        let mut x = SharedTensor::<T>::new(backend.device(), &vec![n]).unwrap();
        let values: &[T] = &::std::iter::repeat(val).take(x.capacity()).collect::<Vec<T>>();
        write_to_memory(x.get_mut(backend.device()).unwrap(), values);
        x
    }

    #[test]
    fn use_cuda_memory_for_asum() {
        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up input
        let n = 20i32;
        let val = 2f32;
        let mut x = filled_tensor(&native, n as usize, val);
        x.add_device(cuda.device()).unwrap();
        x.sync(cuda.device()).unwrap();

        // set up result
        let mut result = SharedTensor::<f32>::new(cuda.device(), &vec![1]).unwrap();
        result.add_device(native.device()).unwrap();
        result.sync(cuda.device()).unwrap();

        {
            let cuda_mem = x.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_result = result.get(cuda.device()).unwrap().as_cuda().unwrap();
            let mut ctx = Context::new().unwrap();
            ctx.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let x_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem.id_c());
                let res_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_result.id_c());
                API::ffi_sasum(*ctx.id_c(), n, x_addr, 1, res_addr).unwrap();
           }
       }

       result.sync(native.device()).unwrap();
       let native_res = result.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[40f32], native_res.as_slice::<f32>());
    }

    #[test]
    fn use_cuda_memory_for_axpy() {
        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up alpha
        let mut alpha = filled_tensor(&native, 1, 1.5f32);
        alpha.add_device(cuda.device()).unwrap();
        alpha.sync(cuda.device()).unwrap();

        // set up x
        let n = 5i32;
        let val = 2f32;
        let mut x = filled_tensor(&native, n as usize, val);
        x.add_device(cuda.device()).unwrap();
        x.sync(cuda.device()).unwrap();

        // set up y
        let val = 4f32;
        let mut y = filled_tensor(&native, n as usize, val);
        y.add_device(cuda.device()).unwrap();
        y.sync(cuda.device()).unwrap();

        {
            let cuda_mem_alpha = alpha.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_x = x.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_y = y.get(cuda.device()).unwrap().as_cuda().unwrap();
            let mut ctx = Context::new().unwrap();
            ctx.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let alpha_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_alpha.id_c());
                let x_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_x.id_c());
                let y_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_y.id_c());
                API::ffi_saxpy(*ctx.id_c(), n, alpha_addr, x_addr, 1, y_addr, 1).unwrap();
           }
       }

       y.sync(native.device()).unwrap();
       let native_y = y.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[7f32, 7f32, 7f32, 7f32, 7f32], native_y.as_slice::<f32>());
    }

    #[test]
    fn use_cuda_memory_for_copy() {
        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up x
        let n = 5i32;
        let val = 2f32;
        let mut x = filled_tensor(&native, n as usize, val);
        x.add_device(cuda.device()).unwrap();
        x.sync(cuda.device()).unwrap();

        // set up y
        let val = 4f32;
        let mut y = filled_tensor(&native, n as usize, val);
        y.add_device(cuda.device()).unwrap();
        y.sync(cuda.device()).unwrap();

        {
            let cuda_mem_x = x.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_y = y.get(cuda.device()).unwrap().as_cuda().unwrap();
            let mut ctx = Context::new().unwrap();
            ctx.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let x_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_x.id_c());
                let y_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_y.id_c());
                API::ffi_scopy(*ctx.id_c(), n, x_addr, 1, y_addr, 1).unwrap();
           }
       }

       y.sync(native.device()).unwrap();
       let native_y = y.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[2f32, 2f32, 2f32, 2f32, 2f32], native_y.as_slice::<f32>());
    }

    #[test]
    fn use_cuda_memory_for_dot() {
        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up x
        let n = 5i32;
        let val = 2f32;
        let mut x = filled_tensor(&native, n as usize, val);
        x.add_device(cuda.device()).unwrap();
        x.sync(cuda.device()).unwrap();

        // set up y
        let val = 4f32;
        let mut y = filled_tensor(&native, n as usize, val);
        y.add_device(cuda.device()).unwrap();
        y.sync(cuda.device()).unwrap();

        // set up result
        let mut result = SharedTensor::<f32>::new(cuda.device(), &vec![1]).unwrap();
        result.add_device(native.device()).unwrap();
        result.sync(cuda.device()).unwrap();

        {
            let cuda_mem_x = x.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_y = y.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_result = result.get(cuda.device()).unwrap().as_cuda().unwrap();
            let mut ctx = Context::new().unwrap();
            ctx.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let x_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_x.id_c());
                let y_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_y.id_c());
                let result_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_result.id_c());
                API::ffi_sdot(*ctx.id_c(), n, x_addr, 1, y_addr, 1, result_addr).unwrap();
           }
       }

       result.sync(native.device()).unwrap();
       let native_result = result.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[40f32], native_result.as_slice::<f32>());
    }

    #[test]
    fn use_cuda_memory_for_nrm2() {
        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up x
        let n = 3i32;
        let val = 2f32;
        let mut x = filled_tensor(&native, n as usize, val);
        write_to_memory(x.get_mut(native.device()).unwrap(), &[2f32, 2f32, 1f32]);
        x.add_device(cuda.device()).unwrap();
        x.sync(cuda.device()).unwrap();

        // set up result
        let mut result = SharedTensor::<f32>::new(cuda.device(), &vec![1]).unwrap();
        result.add_device(native.device()).unwrap();
        result.sync(cuda.device()).unwrap();

        {
            let cuda_mem_x = x.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_result = result.get(cuda.device()).unwrap().as_cuda().unwrap();
            let mut ctx = Context::new().unwrap();
            ctx.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let x_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_x.id_c());
                let result_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_result.id_c());
                API::ffi_snrm2(*ctx.id_c(), n, x_addr, 1, result_addr).unwrap();
           }
       }

       result.sync(native.device()).unwrap();
       let native_result = result.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[3f32], native_result.as_slice::<f32>());
    }

    #[test]
    fn use_cuda_memory_for_scal() {
        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up alpha
        let mut alpha = filled_tensor(&native, 1, 2.5f32);
        alpha.add_device(cuda.device()).unwrap();
        alpha.sync(cuda.device()).unwrap();

        // set up x
        let n = 3i32;
        let val = 2f32;
        let mut x = filled_tensor(&native, n as usize, val);
        x.add_device(cuda.device()).unwrap();
        x.sync(cuda.device()).unwrap();

        {
            let cuda_mem_alpha = alpha.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_x = x.get(cuda.device()).unwrap().as_cuda().unwrap();
            let mut ctx = Context::new().unwrap();
            ctx.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let alpha_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_alpha.id_c());
                let x_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_x.id_c());
                API::ffi_sscal(*ctx.id_c(), n, alpha_addr, x_addr, 1).unwrap();
           }
       }

       x.sync(native.device()).unwrap();
       let native_x = x.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[5f32, 5f32, 5f32], native_x.as_slice::<f32>());
    }

    #[test]
    fn use_cuda_memory_for_swap() {
        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up x
        let n = 5i32;
        let val = 2f32;
        let mut x = filled_tensor(&native, n as usize, val);
        x.add_device(cuda.device()).unwrap();
        x.sync(cuda.device()).unwrap();

        // set up y
        let val = 4f32;
        let mut y = filled_tensor(&native, n as usize, val);
        y.add_device(cuda.device()).unwrap();
        y.sync(cuda.device()).unwrap();

        {
            let cuda_mem_x = x.get(cuda.device()).unwrap().as_cuda().unwrap();
            let cuda_mem_y = y.get(cuda.device()).unwrap().as_cuda().unwrap();
            let mut ctx = Context::new().unwrap();
            ctx.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let x_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_x.id_c());
                let y_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_y.id_c());
                API::ffi_sswap(*ctx.id_c(), n, x_addr, 1, y_addr, 1).unwrap();
           }
       }

       x.sync(native.device()).unwrap();
       let native_x = x.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[4f32, 4f32, 4f32, 4f32, 4f32], native_x.as_slice::<f32>());

       y.sync(native.device()).unwrap();
       let native_y = y.get(native.device()).unwrap().as_native().unwrap();
       assert_eq!(&[2f32, 2f32, 2f32, 2f32, 2f32], native_y.as_slice::<f32>());
    }
}
