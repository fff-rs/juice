//! Provides informations about the software system, such as OpenCL, CUDA, that contains the set of
//! components to support [hardwares][hardware] with kernel execution.
//! [hardware]: ../hardware/index.html
//!
//!

use framework::{IFramework, FrameworkError};
use hardware::{HardwareType, IHardware};
use device::IDevice;
use self::hardware::Hardware;
pub use self::device::Cpu;

pub mod device;
pub mod flatbox;
pub mod hardware;

#[derive(Debug, Clone)]
/// Provides the Native framework.
///
/// Native means host CPU only. The setup one relies on by default.
pub struct Native {
    hardwares: Vec<Hardware>
}

/// Provides the Native framework trait for explicit Backend behaviour.
///
/// You usually would not need to care about this trait.
pub trait INative {}

impl INative for Native {}

impl IFramework for Native {
    type H = Hardware;
    type D = Cpu;
    const ID: &'static str = "NATIVE";

    fn new() -> Native {
        match Native::load_hardwares() {
            Ok(hardwares) => Native { hardwares: hardwares },
            Err(err) => panic!(err)
        }
    }

    fn load_hardwares() -> Result<Vec<Hardware>, FrameworkError> {
        let cpu = Hardware::new(1)
            .set_name(Some(String::from("Host CPU")))
            .set_hardware_type(Some(HardwareType::CPU))
            .set_compute_units(Some(1))
            .build();
        Ok(vec!(cpu))
    }

    fn hardwares(&self) -> Vec<Hardware> {
        self.hardwares.clone()
    }

    fn new_device(&self, devices: Vec<Hardware>) -> Result<Cpu, FrameworkError> {
        Ok(Cpu::new(devices.to_vec()))
    }
}
