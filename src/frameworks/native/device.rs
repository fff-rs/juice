//! Provides a hardware aka. the host CPU.

use device::{IDevice, DeviceType, IDeviceSyncOut};
use device::Error as DeviceError;
use memory::MemoryType;
use super::hardware::Hardware;
use super::Error;
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

    fn id(&self) -> &isize {
        static ID: isize = 0;
        &ID
    }

    fn hardwares(&self) -> Vec<Hardware> {
        self.hardwares.clone()
    }

    fn alloc_memory(&self, size: u64) -> Result<FlatBox, DeviceError> {
        let vec: Vec<u8> = vec![0; size as usize];
        let bx: Box<[u8]> = vec.into_boxed_slice();
        Ok(FlatBox::from_box(bx))
    }

    fn sync_in(&self, source: &DeviceType, source_data: &MemoryType, dest_data: &mut FlatBox) -> Result<(), DeviceError> {
        match source {
            &DeviceType::Native(ref cpu) => unimplemented!(),
            &DeviceType::OpenCL(ref context) => unimplemented!(),
            #[cfg(feature = "cuda")]
            &DeviceType::Cuda(ref context) => {
                match source_data.as_cuda() {
                    Some(mut h_mem) => Ok(try!(context.sync_out(&DeviceType::Native(self.clone()), &h_mem, dest_data))),
                    None => Err(DeviceError::Native(Error::Memory("Expected CUDA Memory")))
                }
            },
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
