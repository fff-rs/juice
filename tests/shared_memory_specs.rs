extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod shared_memory_spec {

    use co::framework::IFramework;
    use co::frameworks::Native;
    use co::frameworks::Cuda;
    use co::memory::MemoryType;
    use co::shared_memory::*;

    #[test]
    fn it_creates_new_shared_memory_for_native() {
        let ntv = Native::new();
        let cpu = ntv.new_device(ntv.hardwares()).unwrap();
        let shared_data = &mut SharedMemory::<f32>::new(&cpu, 10).unwrap();
        if let &MemoryType::Native(ref dat) = shared_data.get(&cpu).unwrap() {
            let data = dat.as_slice::<f32>();
            assert_eq!(10, data.len());
        }
    }

    #[test]
    fn it_creates_new_shared_memory_for_cuda() {
        let ntv = Cuda::new();
        let device = ntv.new_device(ntv.hardwares()[0..1].to_vec()).unwrap();
        let shared_data = &mut SharedMemory::<f32>::new(&device, 10).unwrap();
        match shared_data.get(&device) {
            Some(&MemoryType::Cuda(_)) => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    fn it_has_correct_latest_device() {
        let ntv = Native::new();
        let cpu_dev = ntv.new_device(ntv.hardwares()).unwrap();
        let shared_data = &mut SharedMemory::<f32>::new(&cpu_dev, 10).unwrap();
        assert_eq!(&cpu_dev, shared_data.latest_device());
    }
}
