use crate::ffi::*;
use crate::{API, Error};
use super::Context;
use super::PointerMode;
use lazy_static::lazy_static;
use log::debug;
use std::collections::HashSet;
use std::convert::AsRef;
use std::convert::TryFrom;
use std::ptr;
use std::ptr::NonNull;
use std::sync::{Mutex,Arc};



// TODO:
// extract the cookie tracking into a separate crate
// which provides a better API than this
//
// usecases:
//  * cudaMalloc / cudaFree
//  * cublasContext_new / _destroy
//  * cudnnContext_new / _destroy
#[derive(Hash,Eq,PartialEq)]
struct Cookie(NonNull<cublasContext>);

unsafe impl std::marker::Send for Cookie { }

impl Cookie {
    fn as_ptr(&self) -> *mut cublasContext {
        self.0.as_ptr()
    }
}

impl TryFrom<cublasHandle_t> for Cookie {
    type Error = Error;
    fn try_from(handle: *mut cublasContext) -> std::result::Result<Self,Self::Error> {
        if let Some(nn) = NonNull::new(handle) {
            Ok(Cookie(nn))
        } else {
            Err(Error::Unknown("cublasHandle is a nullptr"))
        }
    }
}

lazy_static! {
    static ref TRACKER: Arc<Mutex<HashSet<Cookie>>> =  {
        Arc::new(Mutex::new(HashSet::with_capacity(3)))
    };
}


fn track(handle: cublasHandle_t) {
    let mut guard = TRACKER.as_ref().lock().unwrap();
    let _ = guard.insert(Cookie::try_from(handle as *mut cublasContext).unwrap());
    debug!("Added handle {:?}, total of {}", handle, guard.len());
}


fn untrack(handle: cublasHandle_t) {
    let mut guard = TRACKER.as_ref().lock().unwrap();
    debug!("Removed handle {:?}, total of {}", handle, guard.len());
    let k = Cookie::try_from(handle as *mut cublasContext).unwrap();
    let _ = guard.remove(&k);
}

fn cleanup() {
    let guard = TRACKER.lock().unwrap();
    for handle in guard.iter() {
        unsafe {
            API::ffi_destroy(handle.as_ptr()).unwrap();
        }
    }
}

impl API {
    /// Create a new cuBLAS context, allocating resources on the host and the GPU.
    ///
    /// The returned Context must be provided to future cuBLAS calls.
    /// Creating contexts all the time can lead to performance problems.
    /// Generally one Context per GPU device and configuration is recommended.
    pub fn create() -> Result<Context, Error> {

        let handle = unsafe { API::ffi_create() }?;
        track(handle);
        Ok(Context::from_c(handle))
    }

    /// Destroys the cuBLAS context, freeing its resources.
    ///
    /// Should generally not be called directly.
    /// Automatically called when dropping a Context.
    ///
    /// # Safety
    /// Instructs CUDA to remove the cuBLAS handle, causing any further instructions to fail.
    /// This should be called at the end of using cuBLAS and should ideally be handled by drop
    /// exclusively, and never called by the user.
    pub unsafe fn destroy(context: &mut Context) -> Result<(), Error> {
        let handle = *context.id_c();
        untrack(handle);
        Ok(API::ffi_destroy(handle)?)
    }

    /// Get CUBLAS Version
    pub fn get_version(context: &Context) -> Result<i32, Error> {
        unsafe {
            API::ffi_get_version(*context.id_c())
        }
    }

    unsafe fn ffi_get_version(handle: cublasHandle_t) -> Result<i32, Error> {
        let mut version: i32 = 0;
        let version_ptr: *mut i32 = &mut version;
        match cublasGetVersion_v2(handle, version_ptr) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(version),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::Unknown("Unable to initialise CUBLAS Library")),
            _ => Err(Error::Unknown("Other Unknown Error with CUBLAS Get Version")),
        }
    }

    /// Retrieve the pointer mode for a given cuBLAS context.
    pub fn get_pointer_mode(context: &Context) -> Result<PointerMode, Error> {
        Ok(PointerMode::from_c(
            unsafe { API::ffi_get_pointer_mode(*context.id_c()) }?,
        ))
    }

    /// Set the pointer mode for a given cuBLAS context.
    pub fn set_pointer_mode(context: &mut Context, pointer_mode: PointerMode) -> Result<(), Error> {
        Ok(unsafe {
            API::ffi_set_pointer_mode(*context.id_c(), pointer_mode.as_c())
        }?)
    }

    unsafe fn ffi_create() -> Result<cublasHandle_t, Error> {
        let mut handle: cublasHandle_t = ptr::null_mut();
        match cublasCreate_v2(&mut handle) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(handle),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => Err(Error::ArchMismatch),
            cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => Err(Error::AllocFailed),
            _ => Err(Error::Unknown(
                "Unable to create the cuBLAS context/resources.",
            )),
        }
    }

    unsafe fn ffi_destroy(handle: cublasHandle_t) -> Result<(), Error> {
        match cublasDestroy_v2(handle) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            _ => Err(Error::Unknown(
                "Unable to destroy the CUDA cuDNN context/resources.",
            )),
        }
    }

    unsafe fn ffi_get_pointer_mode(handle: cublasHandle_t) -> Result<cublasPointerMode_t, Error> {
        let pointer_mode = &mut [cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST];
        match cublasGetPointerMode_v2(handle, pointer_mode.as_mut_ptr()) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(pointer_mode[0]),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            _ => Err(Error::Unknown("Unable to get cuBLAS pointer mode.")),
        }
    }

    unsafe fn ffi_set_pointer_mode(
        handle: cublasHandle_t,
        pointer_mode: cublasPointerMode_t,
    ) -> Result<(), Error> {
        match cublasSetPointerMode_v2(handle, pointer_mode) {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => Err(Error::NotInitialized),
            _ => Err(Error::Unknown("Unable to get cuBLAS pointer mode.")),
        }
    }

    // TODO: cublasGetVersion_v2
    // TODO: cublasSetStream_v2
    // TODO: cublasGetStream_v2
    // TODO: cublasGetAtomicsMode
    // TODO: cublasSetAtomicsMode
    // TODO: cublasSetVector
    // TODO: cublasGetVector
    // TODO: cublasSetMatrix
    // TODO: cublasGetMatrix
    // TODO: cublasSetVectorAsync
    // TODO: cublasGetVectorAsync
    // TODO: cublasSetMatrixAsync
    // TODO: cublasGetMatrixAsync
}

#[cfg(test)]
mod test {
    use crate::ffi::cublasPointerMode_t;
    use crate::API;
    use crate::Context;

    #[test]
    #[serial_test::serial]
    fn manual_context_creation() {
        crate::chore::test_setup();

        unsafe {
            let handle = API::ffi_create().unwrap();
            API::ffi_destroy(handle).unwrap();
        }
    }

    #[test]
    #[serial_test::serial]
    fn default_pointer_mode_is_host() {
        crate::chore::test_setup();

        unsafe {
            dbg!("Pointer Mode Test - Initialises New CUBLAS");
            let context = Context::new().unwrap();
            let mode = API::ffi_get_pointer_mode(*context.id_c()).unwrap();
            assert_eq!(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST, mode);
        }
        crate::chore::test_teardown();
    }

    #[test]
    #[serial_test::serial]
    fn can_set_pointer_mode() {
        crate::chore::test_setup();

        unsafe {
            let context = Context::new().unwrap();
            API::ffi_set_pointer_mode(
                *context.id_c(),
                cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            ).unwrap();
            let mode = API::ffi_get_pointer_mode(*context.id_c()).unwrap();
            assert_eq!(cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE, mode);
            API::ffi_set_pointer_mode(
                *context.id_c(),
                cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            ).unwrap();
            let mode2 = API::ffi_get_pointer_mode(*context.id_c()).unwrap();
            assert_eq!(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST, mode2);
        }
        crate::chore::test_teardown();
    }
}
