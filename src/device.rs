//! Provides a representation for one or many ready to use hardwares.
//!
//! Devices are a set of hardwares, which got initialized from the framework, in order that they
//! are ready to receive kernel executions, event processing, memory synchronization, etc. You can
//! turn available hardware into a device, through the [backend][backend].
//!
//! [backend]: ../backend/index.html

use hardware::IHardware;
use memory::{IMemory, MemoryType};
use operation::IOperation;
use std::hash::{Hash, Hasher};
use frameworks::native::device::Cpu;
use frameworks::opencl::context::Context;

/// Specifies Hardware behavior accross frameworks.
pub trait IDevice {
    /// The Hardware representation for this Device.
    type H: IHardware;
    /// The Memory representation for this Device.
    type M: IMemory;
    /// Returns the unique identifier of the Device.
    fn id(&self) -> isize;
    /// Returns the hardwares, which define the Device.
    fn hardwares(&self) -> Vec<Self::H>;
    /// Allocate memory on the Device.
    fn alloc_memory(&self, size: usize) -> Self::M;
    /// Synchronize memory from this Device to `dest_device`.
    fn sync_memory_to(&self, source: &Self::M, dest: &mut MemoryType, dest_device: &DeviceType);
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
/// Container for all known IDevice implementations
pub enum DeviceType {
    /// A native CPU
    Native(Cpu),
    /// A OpenCL Context
    OpenCL(Context),
}
