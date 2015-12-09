//! Provides a Rust wrapper around OpenCL's context.

use device::{IDevice, DeviceType, IDeviceSyncOut};
use device::Error as DeviceError;
use super::api::types as cl;
use super::{API, Error, Device, Queue};
use super::memory::*;
use memory::MemoryType;
use frameworks::native::flatbox::FlatBox;
use std::{ptr, mem};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
/// Defines a OpenCL Context.
pub struct Context {
    id: isize,
    devices: Vec<Device>,
    queue: Option<Queue>
}

impl Context {
    /// Initializes a new OpenCL platform.
    pub fn new(devices: Vec<Device>) -> Result<Context, Error> {
        let callback = unsafe { mem::transmute(ptr::null::<fn()>()) };
        let mut context = Context::from_c(
                        try!(API::create_context(devices.clone(), ptr::null(), callback, ptr::null_mut())),
                        devices.clone());
        // initialize queue
        context.queue_mut();
        Ok(context)
    }

    /// Initializes a new OpenCL platform from its C type.
    pub fn from_c(id: cl::context_id, devices: Vec<Device>) -> Context {
        Context { id: id as isize, devices: devices, queue: None }
    }

    /// Returns Queue for first device.
    pub fn queue(&self) -> Option<&Queue> {
        self.queue.as_ref()
    }

    /// Returns mutable Queue for first device and creates it if it does not exist yet.
    pub fn queue_mut(&mut self) -> &mut Queue {
        if self.queue.is_some() {
            self.queue.as_mut().unwrap()
        } else {
            self.queue = Some(Queue::new(self, &self.hardwares()[0], None).unwrap());
            self.queue.as_mut().unwrap()
        }
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

impl IDeviceSyncOut<FlatBox> for Context {
    type M = Memory;
    fn sync_out(&self, source_data: &Memory, dest_data: &mut FlatBox) -> Result<(), DeviceError> {
        try!(API::read_from_memory(self.queue().unwrap(), source_data, true, 0, dest_data.byte_size(), dest_data.as_mut_slice().as_mut_ptr(), &[]));
        Ok(())
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

    fn alloc_memory(&self, size: usize) -> Result<Memory, DeviceError> {
        Ok(try!(Memory::new(self, size)))
    }

    fn sync_in(&self, source: &DeviceType, source_data: &MemoryType, dest_data: &mut Memory) -> Result<(), DeviceError> {
        match source {
            &DeviceType::Native(_) => {
                match source_data.as_native() {
                    Some(h_mem) => {
                        try!(API::write_to_memory(self.queue().unwrap(), dest_data, true, 0, h_mem.byte_size(), h_mem.as_slice().as_ptr(), &[]));
                        Ok(())
                    }
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
