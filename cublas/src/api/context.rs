use ffi::*;
use ::{API, Error};
use super::PointerMode;

#[derive(Debug, Clone)]
/// Provides a the low-level cuBLAS context.
pub struct Context {
    id: cublasHandle_t,
}

unsafe impl ::std::marker::Sync for Context {}

impl Drop for Context {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        unsafe{ API::destroy(self) };
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
        Context { id: id }
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
