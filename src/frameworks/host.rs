//! Provides informations about the software system, such as OpenCL, CUDA, that contains the set of
//! components to support [devices][device] with kernel execution.
//! [device]: ../device/index.html
//!
//!

use framework::{IFramework, FrameworkError};
use device::{Device, DeviceType};

#[derive(Debug, Copy, Clone)]
/// Provides the OpenCL Framework.
pub struct Host;

impl IFramework for Host {

    const ID: &'static str = "HOST";

    fn new() -> Host {
        Host
    }

    fn load_devices() -> Vec<Device> {
        let device = Device::new(1)
            .set_name(Some(String::from("Host CPU")))
            .set_device_type(Some(DeviceType::CPU))
            .set_compute_units(Some(1))
            .build();
        vec!(device)
    }

}
