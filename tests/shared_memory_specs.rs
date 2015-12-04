extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod shared_memory_spec {

    use co::framework::IFramework;
    use co::frameworks::Native;
    #[cfg(feature = "cuda")]
    use co::frameworks::Cuda;
    use co::memory::MemoryType;
    use co::shared_memory::*;

    fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
        if let &mut MemoryType::Native(ref mut mem) = mem {
            let mut mem_buffer = mem.as_mut_slice::<T>();
            for (index, datum) in data.iter().enumerate() {
                mem_buffer[index] = *datum;
            }
        }
    }

    #[test]
    fn it_creates_new_shared_memory_for_native() {
        let ntv = Native::new();
        let cpu = ntv.new_device(ntv.hardwares()).unwrap();
        let shared_data = &mut SharedMemory::<f32, TensorR1>::new(&cpu, TensorR1::new([10])).unwrap();
        if let &MemoryType::Native(ref dat) = shared_data.get(&cpu).unwrap() {
            let data = dat.as_slice::<f32>();
            assert_eq!(10, data.len());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn it_creates_new_shared_memory_for_cuda() {
        let ntv = Cuda::new();
        let device = ntv.new_device(ntv.hardwares()[0..1].to_vec()).unwrap();
        let shared_data = &mut SharedMemory::<f32, TensorR1>::new(&device, TensorR1::new([10])).unwrap();
        match shared_data.get(&device) {
            Some(&MemoryType::Cuda(_)) => assert!(true),
            _ => assert!(false),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn it_syncs_from_native_to_cuda_and_back() {
        let cu = Cuda::new();
        let nt = Native::new();
        let cu_device = cu.new_device(cu.hardwares()[0..1].to_vec()).unwrap();
        let nt_device = nt.new_device(nt.hardwares()).unwrap();
        let mem = &mut SharedMemory::<f64, TensorR1>::new(&nt_device, TensorR1::new([3])).unwrap();
        write_to_memory(mem.get_mut(&nt_device).unwrap(), &[1, 2, 3]);
        mem.add_device(&cu_device);
        match mem.sync(&cu_device) {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("{:?}", err);
                assert!(false);
            }
        }
        // It has not successfully synced to the device.
        // Not the other way around.
        match mem.sync(&nt_device) {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("{:?}", err);
                assert!(false);
            }
        }
    }

    #[test]
    fn it_has_correct_latest_device() {
        let ntv = Native::new();
        let cpu_dev = ntv.new_device(ntv.hardwares()).unwrap();
        let shared_data = &mut SharedMemory::<f32, TensorR1>::new(&cpu_dev, TensorR1::new([10])).unwrap();
        assert_eq!(&cpu_dev, shared_data.latest_device());
    }
}
