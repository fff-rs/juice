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

use framework::{IFramework, FrameworkError};
pub use self::platform::Platform;
pub use self::context::Context;
pub use self::memory::Memory;
pub use self::queue::Queue;
pub use self::kernel::Kernel;
pub use self::program::Program;
pub use self::device::{Device, DeviceInfo};
pub use self::api::{API, Error};

pub mod device;
pub mod platform;
pub mod context;
pub mod memory;
pub mod queue;
pub mod kernel;
pub mod program;
pub mod libraries;
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
    const ID: &'static str = "OPENCL";

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

    fn load_hardwares() -> Result<Vec<Device>, FrameworkError> {
        let platforms = match API::load_platforms() {
            Ok(p) => p,
            Err(err) => return Err(FrameworkError::OpenCL(err))
        };

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

    fn binary(&self) -> Self::B {
        self.binary.clone()
    }

    /// Creates a new OpenCL context over one or many devices ready for computation.
    ///
    /// Contexts are used by the OpenCL runtime for managing objects such as command-queues,
    /// memory, program and kernel objects and for executing kernels on one or more hardwares
    /// specified in the context.
    fn new_device(&self, hardwares: Vec<Device>) -> Result<Context, FrameworkError> {
        Ok(try!(Context::new(hardwares)))
    }
}
