extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod shared_memory_spec {

    use co::framework::IFramework;
    use co::frameworks::Native;

    use co::device::{IDevice, DeviceType};
    use co::memory::MemoryType;

    use co::shared_memory::*;

    #[test]
    fn it_creates_buffer() {
        let ntv = Native::new();
        let cpu = ntv.new_device(ntv.hardwares()).unwrap();
        let cpu_dev = &mut DeviceType::Native(cpu.clone());
        let shared_data = &mut SharedMemory::<f32>::new(cpu_dev, 10);
        if let &MemoryType::Native(ref dat) = shared_data.get(cpu_dev).unwrap() {
            let data = dat.as_slice::<f32>();
            assert_eq!(10, data.len());
        }
    }
}
