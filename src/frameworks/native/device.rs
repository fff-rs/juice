//! Provides a hardware aka. the host CPU.

use device::{IDevice, DeviceType};
use memory::MemoryType;
use super::hardware::Hardware;
use super::flatbox::FlatBox;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
/// Defines the host CPU Hardware.
///
/// Can later be transformed into a [Collenchyma hardware][hardware].
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

    fn id(&self) -> isize {
        0
    }

    fn hardwares(&self) -> Vec<Hardware> {
        self.hardwares.clone()
    }

    fn alloc_memory(&self, size: usize) -> FlatBox {
        let vec: Vec<u8> = vec![0; size];
        let bx: Box<[u8]> = vec.into_boxed_slice();
        FlatBox::from_box(bx)
    }

    fn sync_memory_to(&self, source: &FlatBox, dest: &mut MemoryType, dest_device: &DeviceType) {
        match dest_device.clone() {
            DeviceType::Native(_) => {},
            DeviceType::OpenCL(ctx) => {
                unimplemented!();
            }
        }
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
