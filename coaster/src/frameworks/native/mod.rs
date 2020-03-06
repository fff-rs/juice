//! Provides informations about the software system, such as OpenCL, CUDA, that contains the set of
//! components to support [hardwares][hardware] with kernel execution.
//! [hardware]: ../hardware/index.html
//!
//!

use crate::framework::IFramework;
use crate::backend::{Backend, IBackend};
use crate::hardware::{HardwareType, IHardware};
use self::hardware::Hardware;
pub use self::device::Cpu;
pub use self::function::Function;
pub use self::binary::Binary;
pub use self::error::Error;
#[cfg(not(feature = "unstable_alloc"))]
pub use self::stable_alloc::allocate_boxed_slice;
#[cfg(feature = "unstable_alloc")]
pub use self::unstable_alloc::allocate_boxed_slice;

pub mod device;
pub mod flatbox;
pub mod hardware;
pub mod function;
pub mod binary;
mod error;
#[cfg(not(feature = "unstable_alloc"))]
mod stable_alloc;
#[cfg(feature = "unstable_alloc")]
mod unstable_alloc;

#[derive(Debug, Clone)]
/// Provides the Native framework.
///
/// Native means host CPU only. The setup one relies on by default.
pub struct Native {
    hardwares: Vec<Hardware>,
    binary: Binary,
}

/// Provides the Native framework trait for explicit Backend behaviour.
///
/// You usually would not need to care about this trait.
pub trait INative {}

impl INative for Native {}

impl IFramework for Native {
    type H = Hardware;
    type D = Cpu;
    type B = Binary;

    fn ID() -> &'static str { "NATIVE" }

    fn new() -> Native {
        match Native::load_hardwares() {
            Ok(hardwares) => Native {
                hardwares,
                binary: Binary::new(),
            },
            Err(err) => panic!(err)
        }
    }

    fn load_hardwares() -> Result<Vec<Hardware>, crate::framework::Error> {
        let cpu = Hardware::new(1)
            .set_name(Some(String::from("Host CPU")))
            .set_hardware_type(Some(HardwareType::CPU))
            .set_compute_units(Some(1))
            .build();
        Ok(vec!(cpu))
    }

    fn hardwares(&self) -> &[Hardware] {
        &self.hardwares
    }

    fn binary(&self) -> &Binary {
        &self.binary
    }

    fn new_device(&self, devices: &[Hardware]) -> Result<Self::D, crate::framework::Error> {
        Ok(Cpu::new(devices.to_vec()))
    }
}

impl IBackend for Backend<Native> {
    type F = Native;

    fn device(&self) -> &Cpu {
        &self.device()
    }
}
