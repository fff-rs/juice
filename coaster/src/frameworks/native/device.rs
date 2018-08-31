//! Provides a hardware aka. the host CPU.
use std::any::Any;
use std::hash::{Hash, Hasher};

use device::{IDevice, MemorySync};
use device::Error as DeviceError;
use super::hardware::Hardware;
use super::flatbox::FlatBox;
use super::allocate_boxed_slice;

#[derive(Debug, Clone)]
/// Defines the host CPU Hardware.
///
/// Can later be transformed into a [Coaster hardware][hardware].
/// [hardware]: ../../hardware/index.html
pub struct Cpu {
    hardwares: Vec<Hardware>
}

impl Cpu {
    /// Initializes a new OpenCL hardware.
    pub fn new(hardwares: Vec<Hardware>) -> Cpu {
        Cpu { hardwares: hardwares }
    }
}

impl IDevice for Cpu {
    type H = Hardware;
    type M = FlatBox;

    fn id(&self) -> &isize {
        static ID: isize = 0;
        &ID
    }

    fn hardwares(&self) -> &Vec<Hardware> {
        &self.hardwares
    }

    fn alloc_memory(&self, size: usize) -> Result<FlatBox, DeviceError> {
        let bx: Box<[u8]> = allocate_boxed_slice(size);
        Ok(FlatBox::from_box(bx))
    }
}

impl MemorySync for Cpu {
    // transfers from/to Cuda and OpenCL are defined on their MemorySync traits
    fn sync_in(&self, my_memory: &mut Any, src_device: &Any, src_memory: &Any)
               -> Result<(), DeviceError> {
        if let Some(_) = src_device.downcast_ref::<Cpu>() {
            let mut my_mem = my_memory.downcast_mut::<FlatBox>().unwrap();
            let src_mem = src_memory.downcast_ref::<FlatBox>().unwrap();
            my_mem.as_mut_slice::<u8>().clone_from_slice(src_mem.as_slice::<u8>());
            return Ok(());
        }

        Err(DeviceError::NoMemorySyncRoute)
    }

    fn sync_out(&self, my_memory: &Any, dst_device: &Any, dst_memory: &mut Any)
                -> Result<(), DeviceError> {
        if let Some(_) = dst_device.downcast_ref::<Cpu>() {
            let my_mem = my_memory.downcast_ref::<FlatBox>().unwrap();
            let mut dst_mem = dst_memory.downcast_mut::<FlatBox>().unwrap();
            dst_mem.as_mut_slice::<u8>().clone_from_slice(my_mem.as_slice::<u8>());
            return Ok(());
        }

        Err(DeviceError::NoMemorySyncRoute)
    }
}

impl PartialEq for Cpu {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Cpu {}

impl Hash for Cpu {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}
