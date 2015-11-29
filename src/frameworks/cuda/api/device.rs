//! Provides the Cuda API with its device functionality.

use libc;
use super::{API, Error};
use frameworks::cuda::{Device, DeviceInfo};
use super::types as cl;
use super::ffi::*;
use std::ptr;
use std::iter::repeat;

impl API {
    /// Returns fully initialized devices for a specific platform.
    ///
    /// Combines the fetching of all device ids and the fetching of the individual device
    /// information.
    pub fn load_devices() -> Result<Vec<Device>, Error> {
        unimplemented!()
    }

    /// Returns a list of available devices for the provided platform.
    pub fn load_device_list() -> Result<Vec<Device>, Error> {
        unimplemented!()
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
        unimplemented!()
    }

    unsafe fn ffi_get_device_info(
        device: cl::device_id,
        param_name: cl::device_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t
    ) -> Result<(), Error> {
        unimplemented!()
    }
}
