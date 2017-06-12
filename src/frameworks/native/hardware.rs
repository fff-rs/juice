//! Provides a hardware aka. the host CPU.

use hardware::{IHardware, HardwareType};

#[derive(Debug, Clone)]
/// Defines the host CPU Hardware.
///
/// Can later be transformed into a [Coaster hardware][hardware].
/// [hardware]: ../../hardware/index.html
pub struct Hardware {
    id: isize,
    name: Option<String>,
    hardware_type: Option<HardwareType>,
    compute_units: Option<isize>,
}

impl Default for Hardware {
    fn default() -> Self {
        Hardware {
            id: -1,
            name: None,
            hardware_type: None,
            compute_units: None,
        }
    }
}

impl Hardware {
    /// Initializes a new OpenCL hardware.
    pub fn new(id: isize) -> Hardware {
        Hardware { id: id, ..Hardware::default() }
    }
}

impl IHardware for Hardware {
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
        self.hardware_type
    }

    fn set_hardware_type(&mut self, hardware_type: Option<HardwareType>) -> Self {
        self.hardware_type = hardware_type;
        self.clone()
    }

    fn compute_units(&self) -> Option<isize> {
        self.compute_units
    }

    fn set_compute_units(&mut self, compute_units: Option<isize>) -> Self {
        self.compute_units = compute_units;
        self.clone()
    }

    fn build(self) -> Hardware {
        Hardware {
            id: self.id(),
            name: self.name(),
            hardware_type: self.hardware_type(),
            compute_units: self.compute_units(),
        }
    }
}
