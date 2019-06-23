//! Provides the OpenCL API with its platform functionality.

use frameworks::opencl::{API, Error};
use frameworks::opencl::Platform;
use super::types as cl;
use super::ffi::*;
use std::ptr;
use std::iter::repeat;
use std::sync::Mutex;

impl API {
    /// Returns a list of available platforms.
    ///
    /// The OpenCL platform layer which implements platform-specific features that allow
    /// applications to
    ///
    /// * query OpenCL devices,
    /// * obtain device configuration information and
    /// * create OpenCL contexts using one or more devices.
    pub fn load_platforms() -> Result<Vec<Platform>, Error> {
        let mut num_platforms = 0;
        // This mutex is used to work around weak OpenCL implementations.
        // On some implementations concurrent calls to clGetPlatformIDs
        // will cause the implantation to return invalid status.
        lazy_static! {
            static ref PLATFORM_MUTEX: Mutex<()> = Mutex::new(());
        }

        let guard = PLATFORM_MUTEX.lock();
        unsafe {API::ffi_get_platform_ids(0, ptr::null_mut(), &mut num_platforms)}?;

        let mut ids: Vec<cl::device_id> = repeat(0 as cl::device_id).take(num_platforms as usize).collect();

        unsafe {API::ffi_get_platform_ids(num_platforms, ids.as_mut_ptr(), &mut num_platforms)}?;

        let _ = guard;

        Ok(ids.iter().map(|id| Platform::from_c(*id) ).collect())
    }

    unsafe fn ffi_get_platform_ids(
        num_entries: cl::uint,
        platforms: *mut cl::platform_id,
        num_platforms: *mut cl::uint
    ) -> Result<(), Error> {
        match clGetPlatformIDs(num_entries, platforms, num_platforms) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_VALUE => Err(Error::InvalidValue("`num_entries` is equal to zero and `platforms` is not NULL or if both `num_platforms` and `platforms` are NULL")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to get platform ids"))
        }
    }
}
