//! Provides the OpenCL API with its context functionality.
//!
//! At Collenchyma device can be understood as a synonym to OpenCL's context.

use libc;
use super::{API, Error};
use frameworks::cuda::Device;
use super::types as cl;
use super::ffi::*;

impl API {
    /// Creates a OpenCL context.
    ///
    /// An OpenCL context is created with one or more devices. Contexts are used by the OpenCL
    /// runtime for managing objects such as command-queues, memory, program and kernel objects
    /// and for executing kernels on one or more devices specified in the context.
    /// An OpenCL context is a synonym to a Collenchyma device.
    pub fn create_context(
        devices: Vec<Device>,
        properties: *const cl::context_properties,
        callback: extern fn (*const libc::c_char, *const libc::c_void, libc::size_t, *mut libc::c_void),
        user_data: *mut libc::c_void
    ) -> Result<cl::context_id, Error> {
        //let mut device_ids: Vec<cl::device_id> = devices.iter().map(|device| device.id_c()).collect();
        //Ok(
        //    try!(
    //            unsafe { API::ffi_create_context(properties, device_ids.len() as u32, device_ids.as_ptr(), callback, user_data) }
    //        )
    //   )
        unimplemented!()
    }

    unsafe fn ffi_create_context(
        properties: *const cl::context_properties,
        num_devices: cl::uint,
        devices: *const cl::device_id,
        pfn_notify: extern fn (*const libc::c_char, *const libc::c_void, libc::size_t, *mut libc::c_void),
        user_data: *mut libc::c_void
    ) -> Result<cl::context_id, Error> {
        unimplemented!()
    }
}
