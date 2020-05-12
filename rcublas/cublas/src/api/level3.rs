use crate::{API, Error};
use super::Context;
use super::Operation;
use crate::ffi::*;

impl API {
    /// Performs a general matrix-matrix multiplication.
    ///
    /// Note: the matrices are expected to be ordered column-major (FORTRAN-style).
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        context: &Context,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: *mut f32,
        a: *mut f32,
        lda: i32,
        b: *mut f32,
        ldb: i32,
        beta: *mut f32,
        c: *mut f32,
        ldc: i32,
    ) -> Result<(), Error> {
        unsafe {
            Self::ffi_sgemm(
                *context.id_c(),
                transa.as_c(),
                transb.as_c(),
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            )
        }
    }

    /// Note: the matrices are expected to be ordered column-major (FORTRAN-style).
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn ffi_sgemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *mut f32,
        a: *mut f32,
        lda: i32,
        b: *mut f32,
        ldb: i32,
        beta: *mut f32,
        c: *mut f32,
        ldc: i32,
    ) -> Result<(), Error> {
        match cublasSgemm_v2(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        ) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => Err(
                Error::InvalidValue("m, n, or k < 0"),
            ),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => Err(Error::ExecutionFailed),
            _ => Err(Error::Unknown("Unable to calculate axpy (alpha * x + y).")),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::ffi::*;
    use crate::API;
    use crate::api::context::Context;
    use crate::api::enums::PointerMode;
    use crate::co::tensor::SharedTensor;
    use crate::chore::*;

    #[test]
    #[serial_test::serial]
    fn use_cuda_memory_for_gemm() {
        test_setup();

        let native = get_native_backend();
        let cuda = get_cuda_backend();

        // set up alpha
        let alpha = filled_tensor(&native, 1, 1f32);

        // set up beta
        let beta = filled_tensor(&native, 1, 0f32);

        // set up a
        let mut a = SharedTensor::<f32>::new(&vec![3, 2]);
        write_to_memory(
            a.write_only(native.device()).unwrap(),
            &[2f32, 5f32, 2f32, 5f32, 2f32, 5f32],
        );

        // set up b
        let mut b = SharedTensor::<f32>::new(&vec![2, 3]);
        write_to_memory(
            b.write_only(native.device()).unwrap(),
            &[4f32, 1f32, 1f32, 4f32, 1f32, 1f32],
        );

        // set up c
        let mut c = SharedTensor::<f32>::new(&vec![3, 3]);

        {
            let transa = cublasOperation_t::CUBLAS_OP_N;
            let transb = cublasOperation_t::CUBLAS_OP_N;
            let m = 3;
            let n = 3;
            let k = 2;
            let lda = 2;
            let ldb = 3;
            let ldc = 3;
            let cuda_mem_alpha = alpha.read(cuda.device()).unwrap();
            let cuda_mem_beta = beta.read(cuda.device()).unwrap();
            let cuda_mem_a = a.read(cuda.device()).unwrap();
            let cuda_mem_b = b.read(cuda.device()).unwrap();
            let cuda_mem_c = c.write_only(cuda.device()).unwrap();
            let mut context = Context::new().unwrap();
            context.set_pointer_mode(PointerMode::Device).unwrap();
            unsafe {
                let alpha_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_alpha.id_c());
                let beta_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_beta.id_c());
                let a_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_a.id_c());
                let b_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_b.id_c());
                let c_addr = ::std::mem::transmute::<u64, *mut f32>(*cuda_mem_c.id_c());
                API::ffi_sgemm(
                    *context.id_c(),
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha_addr,
                    b_addr,
                    ldb,
                    a_addr,
                    lda,
                    beta_addr,
                    c_addr,
                    ldc,
                ).unwrap();
            }
        }

        let native_c = c.read(native.device()).unwrap();
        assert_eq!(
            &[28f32, 7f32, 7f32, 28f32, 7f32, 7f32, 28f32, 7f32, 7f32],
            native_c.as_slice::<f32>()
        );

        test_teardown();
    }
}
