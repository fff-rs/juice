//! Provides a representation for a collection of available compute units e.g. CPUs or GPUs.
//!
//! Hardware can be GPUs, multi-core CPUs or DSPs, Cell/B.E. processor or whatever else
//! is supported by the provided frameworks such as OpenCL, CUDA, etc. The struct holds all
//! important information about the hardware.
//! To execute code on hardware, turn hardware into a [device][device].
//!
//! [device]: ../device/index.html

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
/// Specifies the available Hardware types.
pub enum HardwareType {
    /// CPU devices
    CPU,
    /// GPU devices
    GPU,
    /// Hardware Accelerator devices
    ACCELERATOR,
    /// Used for anything else
    OTHER,
}

/// Specifies Hardware behavior accross frameworks.
pub trait IHardware {
    /// Returns the ID of the Hardware
    fn id(&self) -> isize;

    /// Returns the name of the Hardware
    fn name(&self) -> Option<String>;

    /// Defines the name of the Hardware
    fn set_name(&mut self, name: Option<String>) -> Self;

    /// Returns the device_type of the Hardware
    fn hardware_type(&self) -> Option<HardwareType>;

    /// Defines the hardware_type of the Hardware
    fn set_hardware_type(&mut self, hardware_type: Option<HardwareType>) -> Self;

    /// Returns the compute_units of the Hardware
    fn compute_units(&self) -> Option<isize>;

    /// Defines the compute_units of the Hardware
    fn set_compute_units(&mut self, compute_units: Option<isize>) -> Self;

    /// Build an inmutable Hardware
    fn build(self) -> Self;
}
