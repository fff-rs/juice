//! Provides the Cuda API with its device functionality.

use frameworks::cuda::api::{API, Error};
use frameworks::cuda::{Device, DeviceInfo};
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use super::ffi::*;

impl API {
    /// Returns fully initialized devices available through Cuda.
    ///
    /// Combines the fetching of all device ids and the fetching of the individual device
    /// information.
    pub fn load_devices() -> Result<Vec<Device>, Error> {
        match API::load_device_list() {
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
    pub fn load_device_list() -> Result<Vec<Device>, Error> {
        let mut device_counter = 0;
        try!(unsafe {API::ffi_device_get_count(&mut device_counter)});

        Ok((0..device_counter).collect::<Vec<i32>>().iter().map(|ordinal| {
            let mut device_id: CUdevice = 0;
            unsafe { API::ffi_device_get(&mut device_id, *ordinal) };
            Device::from_isize(device_id as isize)
        }).collect::<Vec<Device>>())
    }

    /// Returns the requested DeviceInfo for the provided device.
    pub fn load_device_info(device: &Device, info: CUdevice_attribute) -> Result<DeviceInfo, Error> {
        match info {
            CUdevice_attribute::CU_DEVICE_NAME => {
                let mut name: [i8; 1024] = [0; 1024];
                try!(unsafe { API::ffi_device_get_name(name.as_mut_ptr(), 1024, device.id_c()) });
                let mut buf: Vec<u8> = vec![];
                // Removes obsolete whitespaces.
                for (i, char) in name.iter().enumerate() {
                    match *char {
                        0 => if i > 1 && name[i-1] != 0 { buf.push(*char as u8) },
                        _ => buf.push(*char as u8)
                    }
                }
                Ok(DeviceInfo::new(buf))
            },
            CUdevice_attribute::CU_DEVICE_MEMORY_TOTAL => {
                unimplemented!()
            },
            _ => {
                let mut value: ::libc::c_int = 0;
                try!(unsafe { API::ffi_device_get_attribute(&mut value, info, device.id_c()) });
                let mut buf = vec![];
                buf.write_i32::<LittleEndian>(value).unwrap();
                Ok(DeviceInfo::new(buf))
            }
        }
    }

    unsafe fn ffi_device_get(
        device: *mut CUdevice,
        ordinal: ::libc::c_int,
    ) -> Result<(), Error> {
        match cuDeviceGet(device, ordinal) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            _ => Err(Error::Unknown(format!("Unable to get Device count."))),
        }
    }

    unsafe fn ffi_device_get_count(
        count: *mut ::libc::c_int
    ) -> Result<(), Error> {
        match cuDeviceGetCount(count) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            _ => Err(Error::Unknown(format!("Unable to get Device count."))),
        }
    }

    unsafe fn ffi_device_get_attribute(
        pi: *mut ::libc::c_int,
        attrib: CUdevice_attribute,
        device: CUdevice,
    ) -> Result<(), Error> {
        match cuDeviceGetAttribute(pi, attrib, device) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            CUresult::CUDA_ERROR_INVALID_DEVICE => Err(Error::InvalidValue(format!("Invalid value for `device` provided."))),
            _ => Err(Error::Unknown(format!("Unable to get device attribute.")))
        }
    }

    unsafe fn ffi_device_get_name(
        name: *mut ::libc::c_char,
        len: ::libc::c_int,
        device: CUdevice,
    ) -> Result<(), Error> {
        match cuDeviceGetName(name, len, device) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            CUresult::CUDA_ERROR_INVALID_DEVICE => Err(Error::InvalidValue(format!("Invalid value for `device` provided."))),
            _ => Err(Error::Unknown(format!("Unable to get device name.")))
        }
    }

    unsafe fn ffi_device_total_mem(
        bytes: *mut size_t,
        device: CUdevice,
    ) -> Result<(), Error> {
        match cuDeviceTotalMem_v2(bytes, device) {
            CUresult::CUDA_SUCCESS => Ok(()),
            CUresult::CUDA_ERROR_DEINITIALIZED => Err(Error::Deinitialized(format!("CUDA got deinitialized."))),
            CUresult::CUDA_ERROR_NOT_INITIALIZED => Err(Error::NotInitialized(format!("CUDA is not initialized."))),
            CUresult::CUDA_ERROR_INVALID_CONTEXT => Err(Error::InvalidContext(format!("No valid context available."))),
            CUresult::CUDA_ERROR_INVALID_VALUE => Err(Error::InvalidValue(format!("Invalid value provided."))),
            CUresult::CUDA_ERROR_INVALID_DEVICE => Err(Error::InvalidValue(format!("Invalid value for `device` provided."))),
            _ => Err(Error::Unknown(format!("Unable to get total mem of device.")))
        }
    }
}
