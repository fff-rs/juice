extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod device_spec {

    use co::device::{Device, DeviceType};

    #[test]
    fn it_works() {
        Device::new(42)
            .set_device_type(Some(DeviceType::GPU))
            .set_name(Some(String::from("Test Device")))
            .set_compute_units(Some(450))
            .build();
    }

    #[test]
    fn it_returns_correct_id() {
        let device = Device::new(42);
        assert_eq!(device.id(), 42);
    }

    #[test]
    fn it_sets_device_type() {
        let device = Device::new(42)
            .set_device_type(Some(DeviceType::CPU))
            .build();

        assert!(match device.device_type() {
            Some(DeviceType::CPU) => true,
            _ => false
        });
    }

    #[test]
    fn it_sets_name() {
        let device = Device::new(42)
            .set_name(Some(String::from("Test Device")))
            .build();

        let string = String::from("Test Device");
        assert!(match device.name() {
            Some(string) => true,
            _ => false
        });
    }

    #[test]
    fn it_sets_compute_units() {
        let device = Device::new(42)
            .set_compute_units(Some(400))
            .build();

        assert!(match device.compute_units() {
            Some(400) => true,
            _ => false
        });
    }

}
