//! Provides a Rust wrapper around OpenCL's context.

use device::{IDevice, DeviceType};
use super::api::types as cl;
use frameworks::opencl::{API, Error, Device};
use super::memory::*;
use memory::MemoryType;
use std::{ptr, mem};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
/// Defines a OpenCL Context.
pub struct Context {
    id: isize,
    devices: Vec<Device>,
}

impl Context {
    /// Initializes a new OpenCL platform.
    pub fn new(devices: Vec<Device>) -> Result<Context, Error> {
        let callback = unsafe { mem::transmute(ptr::null::<fn()>()) };
        Ok(
            Context::from_c(
                try!(API::create_context(devices.clone(), ptr::null(), callback, ptr::null_mut())),
                devices.clone()
            )
        )
    }

    /// Initializes a new OpenCL platform from its C type.
    pub fn from_c(id: cl::context_id, devices: Vec<Device>) -> Context {
        Context { id: id as isize, devices: devices }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::context_id {
        self.id as cl::context_id
    }
}

impl IDevice for Context {
    type H = Device;
    type M = Memory;

    fn id(&self) -> isize {
        self.id
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
            DeviceType::OpenCL(_) => {},
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
