extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
#[cfg(feature = "opencl")]
mod hardware_spec {
    use co::hardware::{IHardware, HardwareType};
    use co::frameworks::opencl::Device;

    #[test]
    fn it_works() {
        Device::from_isize(42)
            .set_hardware_type(Some(HardwareType::GPU))
            .set_name(Some(String::from("Test Hardware")))
            .set_compute_units(Some(450))
            .build();
    }

    #[test]
    fn it_returns_correct_id() {
        let hardware = Device::from_isize(42);
        assert_eq!(hardware.id(), 42);
    }

    #[test]
    fn it_sets_hardware_type() {
        let hardware = Device::from_isize(42)
            .set_hardware_type(Some(HardwareType::CPU))
            .build();

        assert!(match hardware.hardware_type() {
            Some(HardwareType::CPU) => true,
            _ => false
        });
    }

    #[test]
    fn it_sets_name() {
        let hardware = Device::from_isize(42)
            .set_name(Some(String::from("Test Hardware")))
            .build();

        let string = String::from("Test Hardware");
        assert!(match hardware.name() {
            Some(string) => true,
            _ => false
        });
    }

    #[test]
    fn it_sets_compute_units() {
        let hardware = Device::from_isize(42)
            .set_compute_units(Some(400))
            .build();

        assert!(match hardware.compute_units() {
            Some(400) => true,
            _ => false
        });
    }
}
