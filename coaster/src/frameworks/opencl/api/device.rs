//! Provides the OpenCL API with its device functionality.
//!
//! At Coaster hardware can be understood as a synonym to OpenCL's device.

use libc;
use frameworks::opencl::{API, Error};
use frameworks::opencl::{Platform, Device, DeviceInfo};
use super::types as cl;
use super::ffi::*;
use std::ptr;
use std::iter::repeat;

impl API {
    /// Returns fully initialized devices for a specific platform.
    ///
    /// Combines the fetching of all device ids and the fetching of the individual device
    /// information.
    pub fn load_devices(platform: &Platform) -> Result<Vec<Device>, Error> {
        match API::load_device_list(platform) {
            Ok(device_list) => {
                Ok(
                    device_list.iter().map(|device| {
                        device.clone()
                            .load_name()
                            .load_device_type()
                            .load_compute_units()
                    }).collect()
                )
            },
            Err(err) => Err(err)
        }
    }

    /// Returns a list of available devices for the provided platform.
    pub fn load_device_list(platform: &Platform) -> Result<Vec<Device>, Error> {
        let mut num_devices = 0;

        // load how many devices are available
        try!(unsafe { API::ffi_get_device_ids(platform.id_c(), cl::CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), (&mut num_devices)) });

        // prepare device id list
        let mut ids: Vec<cl::device_id> = repeat(0 as cl::device_id).take(num_devices as usize).collect();

        // load the specific devices
        try!(unsafe { API::ffi_get_device_ids(platform.id_c(), cl::CL_DEVICE_TYPE_ALL, ids.len() as cl::uint, ids.as_mut_ptr(), ptr::null_mut()) });

        Ok(ids.iter().map(|id| Device::from_c(*id) ).collect())
    }

    /// Returns the requested DeviceInfo for the provided device.
    pub fn load_device_info(device: &Device, info: cl::device_info) -> Result<DeviceInfo, Error> {
        let mut size = 0;

        try!(unsafe {API::ffi_get_device_info(device.id_c(), info, 0, ptr::null_mut(), &mut size)});

        let mut buf: Vec<u8> = repeat(0u8).take(size).collect();
        let buf_ptr = buf.as_mut_ptr() as *mut libc::c_void;

        try!(unsafe {API::ffi_get_device_info(device.id_c(), info, size, buf_ptr, ptr::null_mut())});

        Ok(DeviceInfo::new(buf))
    }

    unsafe fn ffi_get_device_ids(
        platform: cl::platform_id,
        device_type: cl::device_type,
        num_entries: cl::uint,
        devices: *mut cl::device_id,
        num_devices: *mut cl::uint
    ) -> Result<(), Error> {
        match clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_PLATFORM => Err(Error::InvalidPlatform("`platform` is not a valid platform")),
            cl::Status::INVALID_DEVICE_TYPE => Err(Error::InvalidDeviceType("`device type` is not a valid device type")),
            cl::Status::DEVICE_NOT_FOUND => Err(Error::DeviceNotFound("no devices for `device type` found")),
            cl::Status::OUT_OF_RESOURCES => Err(Error::OutOfResources("Failure to allocate resources on the device")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to get device ids"))
        }
    }

    unsafe fn ffi_get_device_info(
        device: cl::device_id,
        param_name: cl::device_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t
    ) -> Result<(), Error> {
        match clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_VALUE => Err(Error::InvalidValue("`param_name` is not one of the supported values")),
            cl::Status::OUT_OF_RESOURCES => Err(Error::OutOfResources("Failure to allocate resources on the device")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to get device info string length"))
        }
    }
}
