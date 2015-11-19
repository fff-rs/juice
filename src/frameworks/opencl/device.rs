//! Provides a Rust wrapper around OpenCL's device.

use hardware::{IHardware, HardwareType};
use frameworks::opencl::OpenCL;
use super::api::types as cl;
use super::api::API;
use std::io::Cursor;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};

#[derive(Debug, Clone)]
/// Defines a OpenCL Device.
///
/// Can later be transformed into a [Collenchyma hardware][hardware].
/// [hardware]: ../../hardware/index.html
pub struct Device {
    id: isize,
    name: Option<String>,
    device_type: Option<HardwareType>,
    compute_units: Option<isize>,
}

impl Default for Device {
    fn default() -> Self {
        Device {
            id: -1,
            name: None,
            device_type: None,
            compute_units: None,
        }
    }
}

impl Device {
    /// Initializes a new OpenCL device.
    pub fn from_isize(id: isize) -> Device {
        Device { id: id, ..Device::default() }
    }

    /// Initializes a new OpenCL device from its C type.
    pub fn from_c(id: cl::device_id) -> Device {
        unsafe { Device { id: id as isize, ..Device::default() } }
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::device_id {
        self.id as cl::device_id
    }

    /// Loads the name of the device via a foreign OpenCL call.
    pub fn load_name(&mut self) -> Self {
        self.name = match unsafe { API::load_device_info(self, cl::CL_DEVICE_NAME) } {
            Ok(result) => Some(result.to_string()),
            Err(err) => None
        };
        self.clone()
    }

    /// Loads the device type via a foreign OpenCL call.
    pub fn load_device_type(&mut self) -> Self {
        self.device_type = match unsafe { API::load_device_info(self, cl::CL_DEVICE_TYPE) } {
            Ok(result) => {
                let device_type = result.to_device_type();
                match device_type {
                    cl::CL_DEVICE_TYPE_CPU => Some(HardwareType::CPU),
                    cl::CL_DEVICE_TYPE_GPU => Some(HardwareType::GPU),
                    cl::CL_DEVICE_TYPE_ACCELERATOR => Some(HardwareType::ACCELERATOR),
                    cl::CL_DEVICE_TYPE_DEFAULT => Some(HardwareType::OTHER),
                    cl::CL_DEVICE_TYPE_CUSTOM => Some(HardwareType::OTHER),
                    _ => None
                }
            },
            Err(err) => None
        };
        self.clone()
    }

    /// Loads the compute units of the device via a foreign OpenCL call.
    pub fn load_compute_units(&mut self) -> Self {
        self.compute_units = match unsafe { API::load_device_info(self, cl::CL_DEVICE_MAX_COMPUTE_UNITS) } {
            Ok(result) => Some(result.to_isize()),
            Err(err) => None
        };
        self.clone()
    }
}

impl IHardware for Device {
    fn id(&self) -> isize {
        self.id
    }

    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    fn set_name(&mut self, name: Option<String>) -> Self {
        self.name = name;
        self.clone()
    }

    fn hardware_type(&self) -> Option<HardwareType> {
        self.device_type
    }

    fn set_hardware_type(&mut self, hardware_type: Option<HardwareType>) -> Self {
        self.device_type = hardware_type;
        self.clone()
    }

    fn compute_units(&self) -> Option<isize> {
        self.compute_units
    }

    fn set_compute_units(&mut self, compute_units: Option<isize>) -> Self {
        self.compute_units = compute_units;
        self.clone()
    }

    #[allow(missing_docs)]
    fn build(self) -> Device {
        Device {
            id: self.id(),
            name: self.name(),
            device_type: self.hardware_type(),
            compute_units: self.compute_units(),
        }
    }
}

#[derive(Debug, Clone)]
/// Defines a generic DeviceInfo container.
///
/// Can be used to transform the info to different outputs.
pub struct DeviceInfo {
    info: Vec<u8>,
}

impl DeviceInfo {

    /// Initializes a new Device Info
    pub fn new(info: Vec<u8>) -> DeviceInfo {
        DeviceInfo { info: info }
    }

    #[allow(missing_docs)]
    pub fn to_string(self) -> String {
        unsafe { String::from_utf8_unchecked(self.info) }
    }

    #[allow(missing_docs)]
    pub fn to_device_type(self) -> cl::device_type {
        let mut bytes = Cursor::new(&self.info);
        bytes.read_u64::<LittleEndian>().unwrap()
    }

    #[allow(missing_docs)]
    pub fn to_isize(self) -> isize {
        let mut bytes = Cursor::new(&self.info);
        bytes.read_u32::<LittleEndian>().unwrap() as isize
    }
}
