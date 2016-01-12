//! Provides a Rust wrapper around Cuda's context.

use device::{IDevice, DeviceType, IDeviceSyncOut};
use device::Error as DeviceError;
use super::api::DriverFFI;
use super::{Driver, DriverError, Device};
use super::memory::*;
#[cfg(feature = "native")]
use frameworks::native::flatbox::FlatBox;
use memory::MemoryType;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[derive(Debug, Clone)]
/// Defines a Cuda Context.
pub struct Context {
    id: Rc<isize>,
    devices: Vec<Device>,
}

impl Drop for Context {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        let id_c = self.id_c().clone();
        if let Some(_) = Rc::get_mut(&mut self.id) {
            Driver::destroy_context(id_c);
        }
    }
}

impl Context {
    /// Initializes a new Cuda context.
    pub fn new(devices: Device) -> Result<Context, DriverError> {
        Ok(
            Context::from_c(
                try!(Driver::create_context(devices.clone())),
                vec!(devices.clone())
            )
        )
    }

    /// Initializes a new Cuda platform from its C type.
    pub fn from_c(id: DriverFFI::CUcontext, devices: Vec<Device>) -> Context {
        Context {
            id: Rc::new(id as isize),
            devices: devices
        }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        *self.id
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> DriverFFI::CUcontext {
        *self.id as DriverFFI::CUcontext
    }
}

#[cfg(feature = "native")]
impl IDeviceSyncOut<FlatBox> for Context {
    type M = Memory;
    fn sync_out(&self, source_data: &Memory, dest_data: &mut FlatBox) -> Result<(), DeviceError> {
        Ok(try!(Driver::mem_cpy_d_to_h(source_data, dest_data)))
    }
}

impl IDevice for Context {
    type H = Device;
    type M = Memory;

    fn id(&self) -> &isize {
        &self.id
    }

    fn hardwares(&self) -> &Vec<Device> {
        &self.devices
    }

    fn alloc_memory(&self, size: DriverFFI::size_t) -> Result<Memory, DeviceError> {
        Ok(try!(Driver::mem_alloc(size)))
    }

    fn sync_in(&self, source: &DeviceType, source_data: &MemoryType, dest_data: &mut Memory) -> Result<(), DeviceError> {
        match source {
            #[cfg(feature = "native")]
            &DeviceType::Native(_) => {
                match source_data.as_native() {
                    Some(h_mem) => Ok(try!(Driver::mem_cpy_h_to_d(h_mem, dest_data))),
                    None => unimplemented!()
                }
            },
            _ => unimplemented!()
        }
    }
}

impl PartialEq for Context {
    fn eq(&self, other: &Self) -> bool {
        self.hardwares() == other.hardwares()
    }
}

impl Eq for Context {}

impl Hash for Context {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
