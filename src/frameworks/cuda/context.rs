//! Provides a Rust wrapper around Cuda's context.

use device::{IDevice, DeviceType, IDeviceSyncOut};
use device::Error as DeviceError;
use super::api::DriverFFI;
use super::{Driver, DriverError, Device};
use super::memory::*;
use frameworks::native::flatbox::FlatBox;
use memory::{MemoryType, IMemory};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
/// Defines a Cuda Context.
pub struct Context {
    id: isize,
    devices: Vec<Device>,
}

impl Drop for Context {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        // Produces Segfaults at tests for 50% at a time.
        // Maybe because CUDA context's are linked to a CPU thread?
        //API::destroy_context(self);
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
        Context { id: id as isize, devices: devices }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> DriverFFI::CUcontext {
        self.id as DriverFFI::CUcontext
    }
}

impl IDeviceSyncOut<FlatBox> for Context {
    type M = Memory;
    fn sync_out(&self, dest: &DeviceType, source_data: &Memory, dest_data: &mut FlatBox) -> Result<(), DeviceError> {
        Ok(try!(Driver::mem_cpy_d_to_h(source_data, dest_data)))
    }
}

impl IDevice for Context {
    type H = Device;
    type M = Memory;

    fn id(&self) -> &isize {
        &self.id
    }

    fn hardwares(&self) -> Vec<Device> {
        self.devices.clone()
    }

    fn alloc_memory(&self, size: u64) -> Result<Memory, DeviceError> {
        Ok(try!(Driver::mem_alloc(size)))
    }

    fn sync_in(&self, source: &DeviceType, source_data: &MemoryType, dest_data: &mut Memory) -> Result<(), DeviceError> {
        match source {
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
        self.id() == other.id()
    }
}

impl Eq for Context {}

impl Hash for Context {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
