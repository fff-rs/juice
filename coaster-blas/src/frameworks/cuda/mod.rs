//! Provides BLAS for a CUDA backend.
#![allow(missing_docs)]
use crate::cublas;
use crate::plugin::*;
use crate::transpose::Transpose;
use coaster::backend::Backend;
use coaster::frameworks::cuda::Cuda;
use coaster::plugin::Error as PluginError;
use coaster::tensor::{ITensorDesc, SharedTensor};
use std::sync::Arc;
use spin::Mutex;

#[macro_use]
pub mod helper;

// TODO Arc<Mutex<..>> kills performance
// but inthe light of crashing unit tests
// this is a good enough solution for the time
// being since we should never end up spinning
// the lock - all uses within juice are
// single threaded given currently existing layers.
// According to the nvidia documentation multiple contexts
// should not be a problem, bun in practice this crashes
// often.
lazy_static! {
    static ref CONTEXT: Arc<Mutex<cublas::Context>> = {
        let mut context = cublas::Context::new().unwrap();
        context
            .set_pointer_mode(cublas::api::PointerMode::Device)
            .unwrap();
        Arc::new(Mutex::new(context))
    };
}

fn cuda_blas_ctx() -> Arc<Mutex<cublas::Context>> {
    CONTEXT.clone()
}

impl Asum<f32> for Backend<Cuda> {
    iblas_asum_for_cuda!(f32);
}

impl Axpy<f32> for Backend<Cuda> {
    iblas_axpy_for_cuda!(f32);
}

impl Copy<f32> for Backend<Cuda> {
    iblas_copy_for_cuda!(f32);
}

impl Dot<f32> for Backend<Cuda> {
    iblas_dot_for_cuda!(f32);
}

impl Nrm2<f32> for Backend<Cuda> {
    iblas_nrm2_for_cuda!(f32);
}

impl Scal<f32> for Backend<Cuda> {
    iblas_scal_for_cuda!(f32);
}

impl Swap<f32> for Backend<Cuda> {
    iblas_swap_for_cuda!(f32);
}

impl Gbmv<f32> for Backend<Cuda> {
    iblas_gbmv_for_cuda!(f32);
}

impl Gemm<f32> for Backend<Cuda> {
    iblas_gemm_for_cuda!(f32);
}
