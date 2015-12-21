//! Provides informations about the software system, such as OpenCL, CUDA, that contains the set of
//! components to support [hardwares][hardware] with kernel execution.
//! [hardware]: ../hardware/index.html
//!
//! ## Ontology
//!
//! OpenCL device -> Hardware  
//! OpenCL context -> Device

#[link(name = "OpenCL")]
#[cfg(target_os = "linux")]
extern { }

use backend::{Backend, IBackend};
use framework::IFramework;
use device::DeviceType;
pub use self::platform::Platform;
pub use self::context::Context;
pub use self::memory::{Memory, MemoryFlags};
pub use self::queue::{Queue, QueueFlags};
pub use self::event::Event;
pub use self::kernel::Kernel;
pub use self::program::Program;
pub use self::device::{Device, DeviceInfo};
pub use self::api::{API, Error};

pub mod device;
pub mod platform;
pub mod context;
pub mod memory;
pub mod queue;
pub mod event;
pub mod kernel;
pub mod program;
mod api;

#[derive(Debug, Clone)]
/// Provides the OpenCL Framework.
pub struct OpenCL {
    hardwares: Vec<Device>,
    binary: Program,
}

/// Provides the OpenCL framework trait for explicit Backend behaviour.
///
/// You usually would not need to care about this trait.
pub trait IOpenCL {}

impl IOpenCL for OpenCL {}

impl IFramework for OpenCL {
    type H = Device;
    type D = Context;
    type B = Program;

    fn ID() -> &'static str { "OPENCL" }

    fn new() -> OpenCL {
        match OpenCL::load_hardwares() {
            Ok(hardwares) => {
                OpenCL {
                    hardwares: hardwares,
                    binary: Program::from_isize(1)
                }
            },
            Err(err) => panic!(err)
        }
    }

    fn load_hardwares() -> Result<Vec<Device>, ::framework::Error> {
        let platforms = try!(API::load_platforms());

        let mut hardware_container: Vec<Device> = vec!();
        for platform in &platforms {
            if let Ok(hardwares) = API::load_devices(platform) {
                hardware_container.append(&mut hardwares.clone())
            }
        }
        Ok(hardware_container)
    }

    fn hardwares(&self) -> Vec<Device> {
        self.hardwares.clone()
    }

    fn binary(&self) -> &Self::B {
        &self.binary
    }

    /// Creates a new OpenCL context over one or many devices ready for computation.
    ///
    /// Contexts are used by the OpenCL runtime for managing objects such as command-queues,
    /// memory, program and kernel objects and for executing kernels on one or more hardwares
    /// specified in the context.
    fn new_device(&self, hardwares: Vec<Device>) -> Result<DeviceType, ::framework::Error> {
        Ok(DeviceType::OpenCL(try!(Context::new(hardwares))))
    }
}

impl IBackend for Backend<OpenCL> {
    type F = OpenCL;

    fn device(&self) -> &DeviceType {
        &self.device()
    }
}
