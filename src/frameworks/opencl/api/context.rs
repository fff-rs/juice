//! Provides the OpenCL API with its context functionality.
//!
//! At Coaster device can be understood as a synonym to OpenCL's context.

use libc;
use frameworks::opencl::{API, Device, Error, Platform};
use frameworks::opencl::context::{ContextInfo,ContextInfoQuery,ContextProperties};
use super::types as cl;
use super::ffi::*;
use std::ptr;
use std::mem::size_of;
use std;

impl API {
    /// Creates a OpenCL context.
    ///
    /// An OpenCL context is created with one or more devices. Contexts are used by the OpenCL
    /// runtime for managing objects such as command-queues, memory, program and kernel objects
    /// and for executing kernels on one or more devices specified in the context.
    /// An OpenCL context is a synonym to a Coaster device.
    pub fn create_context(
        devices: Vec<Device>,
        properties: *const cl::context_properties,
        callback: extern fn (*const libc::c_char, *const libc::c_void, libc::size_t, *mut libc::c_void),
        user_data: *mut libc::c_void
    ) -> Result<cl::context_id, Error> {
        let device_ids: Vec<cl::device_id> = devices.iter().map(|device| device.id_c()).collect();
        Ok(
            try!(
                unsafe { API::ffi_create_context(properties, device_ids.len() as u32, device_ids.as_ptr(), callback, user_data) }
            )
        )
    }

    /// FFI Creates an OpenCL context.
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

    /// Gets info about one of the available properties of an OpenCL context.
    pub fn get_context_info(
        context: cl::context_id,
        query: ContextInfoQuery ,
    ) -> Result<ContextInfo, Error> {

        let info_name : cl::context_info = match query {
            ContextInfoQuery::ReferenceCount => cl::CL_CONTEXT_REFERENCE_COUNT,
            ContextInfoQuery::NumDevices => cl::CL_CONTEXT_NUM_DEVICES,
            ContextInfoQuery::Properties => cl::CL_CONTEXT_PROPERTIES,
            ContextInfoQuery::Devices => cl::CL_CONTEXT_DEVICES,
        };

        Ok({
            unsafe {
                let mut zero: usize = 0;
                let info_size: *mut usize = &mut zero;
                API::ffi_get_context_info_size(context, info_name, info_size)
                    .and_then(|_| {
                        let mut buffer = vec![0u8; *info_size];
                        let info_ptr: *mut libc::c_void = buffer.as_mut_ptr() as *mut libc::c_void;
                        API::ffi_get_context_info(context,
                                                  info_name,
                                                  *info_size,
                                                  info_ptr)
                        .and_then(|_| {
                        match info_name {
                            cl::CL_CONTEXT_REFERENCE_COUNT => {
                                let reference_count : u32 = *(info_ptr as *mut u32);
                                Ok(ContextInfo::ReferenceCount(reference_count))
                            },
                            cl::CL_CONTEXT_DEVICES => {
                                let len = *info_size / size_of::<cl::uint>();
                                let mut dev_ids : Vec<cl::uint> = Vec::new();
                                let info_ptr : *mut cl::uint = info_ptr as *mut cl::uint;
                                for i in 0..len as isize {
                                    dev_ids.push(*info_ptr.offset(i));
                                }
                                Ok(ContextInfo::Devices(
                                    dev_ids
                                        .iter()
                                        .map(|&id| Device::from_isize(id as isize))
                                        .collect()
                                ))
                            },
                            cl::CL_CONTEXT_NUM_DEVICES => {
                                let device_count : u32 = *(info_ptr as *mut u32);
                                Ok(ContextInfo::NumDevices(device_count))
                            },
                            cl::CL_CONTEXT_PROPERTIES => {
                                let mut v : Vec<ContextProperties> = Vec::new();
                                let mut ptr : *mut u8 = info_ptr as *mut u8;
                                let mut total_decoded: isize = 0;
                                let info_size = *info_size as isize;
                                println!("{:?}", info_size);
                                while total_decoded < info_size {
                                    // get the identifier and advance by identifier size count bytes
                                    let identifier : *mut cl::context_properties = ptr as *mut cl::context_properties;
                                    let identifier = *identifier;
                                    ptr = ptr.offset(std::mem::size_of::<cl::context_properties>() as isize);
                                    // depending on the identifier decode the per identifier payload/argument with the
                                    // corresponding type
                                    match identifier {
                                        cl::CL_CONTEXT_PLATFORM => {
                                            let platform_id : *const cl::platform_id = info_ptr  as *const cl::platform_id;
                                            let platform_id = *platform_id;
                                            let size = std::mem::size_of::<cl::platform_id>() as isize;
                                            total_decoded += size;
                                            ptr = ptr.offset(size);
                                            v.push(ContextProperties::Platform(Platform::from_c(platform_id)));
                                        },
                                        cl::CL_CONTEXT_INTEROP_USER_SYNC => {
                                            let interop_user_sync : *const cl::boolean = info_ptr as *const cl::boolean;
                                            let interop_user_sync = *interop_user_sync == 0;
                                            let size = std::mem::size_of::<cl::boolean>() as isize;
                                            total_decoded += size;
                                            ptr = ptr.offset(size);
                                            v.push(ContextProperties::InteropUserSync(interop_user_sync));
                                        },
                                        0 => {
                                            break;
                                        }
                                        _ => {
                                            return Err(Error::Other("Unknown property"));
                                        }
                                    };
                                }
                                Ok(ContextInfo::Properties(v))
                            }
                            _ => {
                                Err(Error::Other("Unknown property"))
                            }
                        }
                    })
                    })
            }

        }?)
    }
    
    // This function calls clGetContextInfo with the return data pointer set to
    // NULL to find out the needed memory allocation first.
    unsafe fn ffi_get_context_info_size(
        context: cl::context_id,
        param_name: cl::context_info,
        param_value_size_ret: *mut libc::size_t
    ) -> Result<(), Error> {
        match clGetContextInfo(context,
                               param_name,
                               0,
                               ptr::null_mut(),
                               param_value_size_ret) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_CONTEXT => Err(Error::InvalidContext("Invalid context")),
            cl::Status::INVALID_VALUE => Err(Error::InvalidValue("Invalid value")),
            cl::Status::OUT_OF_RESOURCES => Err(Error::OutOfResources("Out of resources")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Out of host memory")),
            _ => Err(Error::Other("Could not determine needed memory to allocate context info."))
        }
    }

    // This function calls clGetContextInfo with the return data pointer set,
    // and the return size pointer set to NULL (since we assume you know before
    // you call this function how much memory you need).
    unsafe fn ffi_get_context_info(
        context: cl::context_id,
        param_name: cl::context_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void) -> Result<(), Error> {
        match clGetContextInfo(context,
                               param_name,
                               param_value_size,
                               param_value,
                               ptr::null_mut()) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_CONTEXT => Err(Error::InvalidContext("Invalid context")),
            cl::Status::INVALID_VALUE => Err(Error::InvalidValue("Invalid value")),
            cl::Status::OUT_OF_RESOURCES => Err(Error::OutOfResources("Out of resources")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Out of host memory")),
            _ => Err(Error::Other("Could not determine needed memory to allocate context info."))
        }
    }
}
