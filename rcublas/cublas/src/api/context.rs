use ffi::*;
use {API, Error};
use super::{Operation, PointerMode};

#[derive(Debug, Clone)]
/// Provides a the low-level cuBLAS context.
pub struct Context {
    id: cublasHandle_t,
}

unsafe impl ::std::marker::Sync for Context {}

impl Drop for Context {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        unsafe { API::destroy(self) };
    }
}

impl Context {
    /// Create a new cuBLAS Context by calling the low-level API.
    ///
    /// Context creation should done as sparely as possible.
    /// It is best to keep a context around as long as possible.
    pub fn new() -> Result<Context, Error> {
        API::create()
    }

    /// Create a new cuBLAS Context from its C type.
    pub fn from_c(id: cublasHandle_t) -> Context {
        Context { id }
    }

    /// Returns the cuBLAS Context as its C type.
    pub fn id_c(&self) -> &cublasHandle_t {
        &self.id
    }

    pub fn pointer_mode(&self) -> Result<PointerMode, Error> {
        API::get_pointer_mode(self)
    }

    pub fn set_pointer_mode(&mut self, pointer_mode: PointerMode) -> Result<(), Error> {
        API::set_pointer_mode(self, pointer_mode)
    }

    // Level 1 operations

    pub fn asum(
        &self,
        x: *mut f32,
        result: *mut f32,
        n: i32,
        stride: Option<i32>,
    ) -> Result<(), Error> {
        API::asum(self, x, result, n, stride)
    }

    pub fn axpy(
        &self,
        alpha: *mut f32,
        x: *mut f32,
        y: *mut f32,
        n: i32,
        stride_x: Option<i32>,
        stride_y: Option<i32>,
    ) -> Result<(), Error> {
        API::axpy(self, alpha, x, y, n, stride_x, stride_y)
    }

    pub fn copy(
        &self,
        x: *mut f32,
        y: *mut f32,
        n: i32,
        stride_x: Option<i32>,
        stride_y: Option<i32>,
    ) -> Result<(), Error> {
        API::copy(self, x, y, n, stride_x, stride_y)
    }

    pub fn dot(
        &self,
        x: *mut f32,
        y: *mut f32,
        result: *mut f32,
        n: i32,
        stride_x: Option<i32>,
        stride_y: Option<i32>,
    ) -> Result<(), Error> {
        API::dot(self, x, y, result, n, stride_x, stride_y)
    }

    pub fn nrm2(
        &self,
        x: *mut f32,
        result: *mut f32,
        n: i32,
        stride_x: Option<i32>,
    ) -> Result<(), Error> {
        API::nrm2(self, x, result, n, stride_x)
    }

    pub fn scal(
        &self,
        alpha: *mut f32,
        x: *mut f32,
        n: i32,
        stride_x: Option<i32>,
    ) -> Result<(), Error> {
        API::scal(self, alpha, x, n, stride_x)
    }

    pub fn swap(
        &self,
        x: *mut f32,
        y: *mut f32,
        n: i32,
        stride_x: Option<i32>,
        stride_y: Option<i32>,
    ) -> Result<(), Error> {
        API::swap(self, x, y, n, stride_x, stride_y)
    }

    // Level 3 operations
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::many_single_char_names)]
    pub fn gemm(
        &self,
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
        API::gemm(
            self,
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
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::super::PointerMode;

    #[test]
    fn create_context() {
        Context::new().unwrap();
    }

    #[test]
    fn default_pointer_mode_is_host() {
        let ctx = Context::new().unwrap();
        let mode = ctx.pointer_mode().unwrap();
        assert_eq!(PointerMode::Host, mode);
    }

    #[test]
    fn can_set_pointer_mode() {
        let mut context = Context::new().unwrap();
        // set to Device
        context.set_pointer_mode(PointerMode::Device).unwrap();
        let mode = context.pointer_mode().unwrap();
        assert_eq!(PointerMode::Device, mode);
        // set to Host
        context.set_pointer_mode(PointerMode::Host).unwrap();
        let mode2 = context.pointer_mode().unwrap();
        assert_eq!(PointerMode::Host, mode2);
    }
}
