//! Provides a Rust wrapper around OpenCL's context.

use device::{IDevice, MemorySync};
use device::Error as DeviceError;
use super::api::types as cl;
use super::{API, Error, Device, Queue};
use super::memory::*;
#[cfg(feature = "native")]
use frameworks::native::flatbox::FlatBox;
#[cfg(feature = "native")]
use frameworks::native::device::Cpu;
use std::any::Any;
use std::{ptr, mem};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
/// Defines a OpenCL Context.
pub struct Context {
    id: isize,
    devices: Vec<Device>,
    queue: Option<Queue>
}

/// OpenCL context info types. Each variant is returned from the same function,
/// `get_context_info`.
#[derive(PartialEq, Debug)]
pub enum ContextInfo {
    /// Number of references to the context currently held.
    ReferenceCount(u32),
    /// Number of devices in the context.
    NumDevices(u32),
    /// The properties the context was configured with.
    ///
    /// These are:
    ///
    /// - CL_CONTEXT_PLATFORM
    /// - CL_CONTEXT_D3D10_DEVICE_KHR
    /// - CL_GL_CONTEXT_KHR
    /// - CL_EGL_CONTEXT_KHR
    /// - CL_GLX_DISPLAY_KHR
    /// - CL_WGL_HDC_KHR
    /// - CL_CGL_SHAREGROUP_KHR
    ///
    /// Only the first property is required--the others may not be there
    /// depending on CL extensions.
    ContextProperties(cl::context_properties),
    /// The devices (IDs) in the context.
    Devices(Vec<Device>)
}

impl Context {
    /// Initializes a new OpenCL platform.
    pub fn new(devices: Vec<Device>) -> Result<Context, Error> {
        let callback = unsafe { mem::transmute(ptr::null::<fn()>()) };
        let mut context = Context::from_c(
                        API::create_context(devices.clone(), ptr::null(), callback, ptr::null_mut())?,
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
        Ok(Memory::new(self, size)?)
    }
}

impl MemorySync for Context {
    fn sync_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any)
               -> Result<(), DeviceError> {
        if let Some(_) = src_device.downcast_ref::<Cpu>() {
            let mut my_mem = my_memory.downcast_mut::<Memory>().unwrap();
            let src_mem = src_memory.downcast_ref::<FlatBox>().unwrap();

            API::write_to_memory(
                self.queue().unwrap(), my_mem, true, 0,
                src_mem.byte_size(), src_mem.as_slice().as_ptr(), &[])?;
            Ok(())
        } else {
            Err(DeviceError::NoMemorySyncRoute)
        }
    }

    fn sync_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any)
                -> Result<(), DeviceError> {
        if let Some(_) = dst_device.downcast_ref::<Cpu>() {
            let my_mem = my_memory.downcast_ref::<Memory>().unwrap();
            let mut dst_mem = dst_memory.downcast_mut::<FlatBox>().unwrap();

            API::read_from_memory(
                self.queue().unwrap(), my_mem, true, 0,
                dst_mem.byte_size(), dst_mem.as_mut_slice().as_mut_ptr(), &[])?;
            Ok(())
        } else {
            Err(DeviceError::NoMemorySyncRoute)
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
