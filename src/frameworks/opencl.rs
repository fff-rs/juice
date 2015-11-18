//! Provides informations about the software system, such as OpenCL, CUDA, that contains the set of
//! components to support [devices][device] with kernel execution.
//! [device]: ../device/index.html
//!
//!

#[link(name = "OpenCL")]
#[cfg(target_os = "linux")]
extern { }

use framework::{IFramework, FrameworkError};
use super::opencl_ffi::*;
use std::sync::{StaticMutex, MUTEX_INIT};
use std::iter::repeat;
use std::ptr;

#[derive(Debug, Copy, Clone)]
/// Provides the OpenCL Framework.
pub struct OpenCL;

///// Defines the OpenCL Framework.
/*
impl IFramework for OpenCL {

    /// Defines the Framework by a Name
    ///
    /// For convention, let the ID be uppercase.<br/>
    /// EXAMPLE: OPENCL
    const ID: &'static str = "OPENCL";

    /// Obtain a list of available platforms.
    //fn get_platform_ids() -> Result<Vec<i32>, FrameworkError>
    {
        let mut num_platforms = 0;

        // This mutex is used to work around weak OpenCL implementations.
        // On some implementations concurrent calls to clGetPlatformIDs
        // will cause the implantation to return invalid status.
        static mut platforms_mutex: StaticMutex = MUTEX_INIT;

        unsafe {
            let guard = platforms_mutex.lock();
            let status = clGetPlatformIDs(0, ptr::null_mut(), (&mut num_platforms));
            match check(status, String::from("Unable to get Platform count.")) {
                Some(e) => return Err(e),
                None => ()
            }

            let mut ids: Vec<cl_device_id> = repeat(0 as cl_device_id)
                .take(num_platforms as usize)
                .collect();

            let status = clGetPlatformIDs(num_platforms, ids.as_mut_ptr(), (&mut num_platforms));
            match check(status, String::from("Unable to get Platforms.")) {
                Some(e) => return Err(e),
                None => ()
            }

            let _ = guard;

            Ok(ids.iter().map(|id| *id as i32 ).collect())
        }
    }

}
*/
