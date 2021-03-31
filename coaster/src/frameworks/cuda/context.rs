//! Provides a Rust wrapper around Cuda's context.

use super::api::DriverFFI;
use super::memory::*;
use super::{Device, Driver, DriverError};
use crate::device::Error as DeviceError;
use crate::device::{IDevice, MemorySync};
use crate::frameworks::native::device::Cpu;
use crate::frameworks::native::flatbox::FlatBox;
use std::any::Any;
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
        let id_c = self.id_c();
        if Rc::get_mut(&mut self.id).is_some() {
            Driver::destroy_context(id_c);
        }
    }
}

impl Context {
    /// Initializes a new Cuda context.
    pub fn new(devices: Device) -> Result<Context, DriverError> {
        Ok(Context::from_c(
            Driver::create_context(devices.clone())?,
            vec![devices],
        ))
    }

    /// Initializes a new Cuda platform from its C type.
    pub fn from_c(id: DriverFFI::CUcontext, devices: Vec<Device>) -> Context {
        Context {
            id: Rc::new(id as isize),
            devices,
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

    /// Synchronize this Context.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        Driver::synchronize_context()
    }
}

// #[cfg(feature = "native")]
// impl IDeviceSyncOut<FlatBox> for Context {
//     type M = Memory;
//     fn sync_out(&self, source_data: &Memory, dest_data: &mut FlatBox) -> Result<(), DeviceError> {
//         Ok(Driver::mem_cpy_d_to_h(source_data, dest_data)?)
//     }
// }

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
        Ok(Driver::mem_alloc(size)?)
    }
}

impl MemorySync for Context {
    fn sync_in(
        &self,
        my_memory: &mut dyn Any,
        src_device: &dyn Any,
        src_memory: &dyn Any,
    ) -> Result<(), DeviceError> {
        if src_device.downcast_ref::<Cpu>().is_some() {
            let my_mem = my_memory.downcast_mut::<Memory>().unwrap();
            let src_mem = src_memory.downcast_ref::<FlatBox>().unwrap();

            Ok(Driver::mem_cpy_h_to_d(src_mem, my_mem)?)
        } else {
            Err(DeviceError::NoMemorySyncRoute)
        }
    }

    fn sync_out(
        &self,
        my_memory: &dyn Any,
        dst_device: &dyn Any,
        dst_memory: &mut dyn Any,
    ) -> Result<(), DeviceError> {
        if dst_device.downcast_ref::<Cpu>().is_some() {
            let my_mem = my_memory.downcast_ref::<Memory>().unwrap();
            let dst_mem = dst_memory.downcast_mut::<FlatBox>().unwrap();
            Ok(Driver::mem_cpy_d_to_h(my_mem, dst_mem)?)
        } else {
            Err(DeviceError::NoMemorySyncRoute)
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
