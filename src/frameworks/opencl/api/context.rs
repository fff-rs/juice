//! Provides the OpenCL API with its context functionality.
//!
//! At Collenchyma device can be understood as a synonym to OpenCL's context.

use libc;
use frameworks::opencl::{API, Error, Device};
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
        let mut device_ids: Vec<cl::device_id> = devices.iter().map(|device| device.id_c()).collect();
        Ok(
            try!(
                unsafe { API::ffi_create_context(properties, device_ids.len() as u32, device_ids.as_ptr(), callback, user_data) }
            )
        )
    }

    unsafe fn ffi_create_context(
        properties: *const cl::context_properties,
        num_devices: cl::uint,
        devices: *const cl::device_id,
        pfn_notify: extern fn (*const libc::c_char, *const libc::c_void, libc::size_t, *mut libc::c_void),
        user_data: *mut libc::c_void
    ) -> Result<cl::context_id, Error> {
        let mut errcode: i32 = 0;
        let context_id = clCreateContext(properties, num_devices, devices, pfn_notify, user_data, &mut errcode);
        match errcode {
            errcode if errcode == cl::Status::SUCCESS as i32 => Ok(context_id),
            errcode if errcode == cl::Status::INVALID_PLATFORM as i32 => Err(Error::InvalidPlatform("properties is NULL and no platform could be selected or if platform value specified in propertiesis not a valid platform.")),
            errcode if errcode == cl::Status::INVALID_PROPERTY as i32 => Err(Error::InvalidProperty("context property name in propertiesis not a supported property name, if the value specified for a supported property name is not valid,or if the same property name is specified more than once.")),
            errcode if errcode == cl::Status::INVALID_VALUE as i32 => Err(Error::InvalidValue("devices is NULL or num_devices is equal to zero or pfn_notify is NULL but user_data is not NULL")),
            errcode if errcode == cl::Status::INVALID_DEVICE as i32 => Err(Error::InvalidDevice("devices contains an invalid device.")),
            errcode if errcode == cl::Status::DEVICE_NOT_AVAILABLE as i32 => Err(Error::DeviceNotAvailable("a device in devices is currently not available even though the device was returned by clGetDeviceIDs.")),
            errcode if errcode == cl::Status::OUT_OF_RESOURCES as i32 => Err(Error::OutOfResources("Failure to allocate resources on the device")),
            errcode if errcode == cl::Status::OUT_OF_HOST_MEMORY as i32 => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to create context"))
        }
    }
}
