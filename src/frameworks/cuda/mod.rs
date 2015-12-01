//! Provides informations about the software system, such as Cuda, CUDA, that contains the set of
//! components to support [hardwares][hardware] with kernel execution.
//! [hardware]: ../hardware/index.html
//!
//! ## Ontology
//!
//! Cuda device -> Hardware
//! Cuda context -> Device

extern { }

use framework::IFramework;
use device::DeviceType;
pub use self::memory::Memory;
pub use self::context::Context;
pub use self::function::Function;
pub use self::module::Module;
pub use self::device::{Device, DeviceInfo};
pub use self::api::{API, Error};

pub mod device;
pub mod context;
pub mod function;
pub mod memory;
pub mod module;
mod api;

#[derive(Debug, Clone)]
/// Provides the Cuda Framework.
pub struct Cuda {
    hardwares: Vec<Device>,
    binary: Module,
}

impl IFramework for Cuda {
    type H = Device;
    type D = Context;
    type B = Module;

    fn ID() -> &'static str { "CUDA" }

    fn new() -> Cuda {
        match Cuda::load_hardwares() {
            Ok(hardwares) => {
                Cuda {
                    hardwares: hardwares,
                    binary: Module::from_isize(1)
                }
            },
            Err(err) => panic!(err)
        }
    }

    fn load_hardwares() -> Result<Vec<Device>, ::framework::Error> {
        unimplemented!()
    }

    fn hardwares(&self) -> Vec<Device> {
        self.hardwares.clone()
    }

    fn binary(&self) -> Self::B {
        self.binary.clone()
    }

    /// Creates a new Cuda device for computation.
    ///
    /// Cuda's device differs from OpenCL's context. Multi device support works different in Cuda.
    /// This function currently suppports only one device, but be a wrapper for multi device support.
    fn new_device(&self, hardwares: Vec<Device>) -> Result<DeviceType, ::framework::Error> {
        unimplemented!()
    }
}
