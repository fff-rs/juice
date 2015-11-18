//! Provides a representation for a collection of compute units e.g. CPUs or GPUs.
//!
//! Devices can be GPUs, multi-core CPUs or DSPs, Cell/B.E. processor or whatever else
//! is supported by the provided frameworks such as OpenCL, CUDA, etc. The struct holds all
//! important information about the device.
//!
//! ## Examples
//!
//! ```
//! extern crate collenchyma as co;
//! use co::device::{Device, DeviceType};
//! # fn main() {
//!
//! let id = 1;
//! let device = Device::new(id)
//!     .set_name(Some(String::from("Super GPU")))
//!     .set_device_type(Some(DeviceType::GPU))
//!     .set_compute_units(Some(450))
//!     .build();
//! # }
//! ```

#[derive(Debug, Copy, Clone)]
/// Specifies the available Device types.
pub enum DeviceType {
    /// CPU devices
    CPU,
    /// GPU devices
    GPU,
    /// Used for anything else
    OTHER
}

#[derive(Debug, Clone)]
/// Defines a Device.
///
/// It implements a builder pattern, that allows to build a Device by chaining setters. At the end
/// call .build, to receive an inmutable Device, which is important, as you do not want to chance
/// your device information after initialization from the Framework.
pub struct Device {
    id: i32,
    name: Option<String>,
    device_type: Option<DeviceType>,
    compute_units: Option<u32>,
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

    /// Initializes a new Device given an ID
    pub fn new(id: i32) -> Device {
        Device { id: id, ..Device::default() }
    }

    /// Returns the ID of the Device
    pub fn id(&self) -> i32 {
        self.id
    }

    /// Returns the name of the Device
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }

    /// Defines the name of the Device
    pub fn set_name(&mut self, name: Option<String>) -> Self {
        self.name = name;
        self.clone()
    }

    /// Returns the device_type of the Device
    pub fn device_type(&self) -> Option<DeviceType> {
        self.device_type
    }

    /// Defines the device_type of the Device
    pub fn set_device_type(&mut self, device_type: Option<DeviceType>) -> Self {
        self.device_type = device_type;
        self.clone()
    }

    /// Returns the compute_units of the Device
    pub fn compute_units(&self) -> Option<u32> {
        self.compute_units
    }

    /// Defines the compute_units of the Device
    pub fn set_compute_units(&mut self, compute_units: Option<u32>) -> Self {
        self.compute_units = compute_units;
        self.clone()
    }

    /// Build an inmutable Device
    pub fn build(self) -> Device {
        Device {
            id: self.id(),
            name: self.name(),
            device_type: self.device_type(),
            compute_units: self.compute_units(),
        }
    }

}
