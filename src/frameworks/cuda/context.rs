//! Provides a Rust wrapper around Cuda's context.

use device::{IDevice, DeviceType};
use super::api::ffi::*;
use super::{API, Error, Device};
use super::memory::*;
use memory::MemoryType;
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
        API::destroy_context(self);
    }
}

impl Context {
    /// Initializes a new Cuda context.
    pub fn new(devices: Device) -> Result<Context, Error> {
        Ok(
            Context::from_c(
                try!(API::create_context(devices.clone())),
                vec!(devices.clone())
            )
        )
    }

    /// Initializes a new Cuda platform from its C type.
    pub fn from_c(id: CUcontext, devices: Vec<Device>) -> Context {
        Context { id: id as isize, devices: devices }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> CUcontext {
        self.id as CUcontext
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

    fn alloc_memory(&self, size: usize) -> Memory {
        unimplemented!();
    }

    fn sync_memory_to(&self, source: &Memory, dest: &mut MemoryType, dest_device: &DeviceType) {
        /*
        let src = Memory::<Vec<u8>>::from_c(source as cl::memory_id);
        match dest_device.clone() {
            DeviceType::Native(cpu) => {
                unimplemented!();
            }
            DeviceType::Cuda(_) => {},
        }
        */
        unimplemented!()
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
