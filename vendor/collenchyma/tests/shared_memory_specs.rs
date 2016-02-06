extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod shared_memory_spec {
    use co::prelude::*;

    fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
        match mem {
            &mut MemoryType::Native(ref mut mem) => {
                let mut mem_buffer = mem.as_mut_slice::<T>();
                for (index, datum) in data.iter().enumerate() {
                    mem_buffer[index] = *datum;
                }
            },
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => assert!(false)
        }
    }

    #[test]
    #[cfg(feature = "native")]
    fn it_creates_new_shared_memory_for_native() {
        let ntv = Native::new();
        let cpu = ntv.new_device(ntv.hardwares()).unwrap();
        let shared_data = &mut SharedTensor::<f32>::new(&cpu, &10).unwrap();
        match shared_data.get(&cpu).unwrap() {
            &MemoryType::Native(ref dat) => {
                let data = dat.as_slice::<f32>();
                assert_eq!(10, data.len());
            },
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => assert!(false)
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn it_creates_new_shared_memory_for_cuda() {
        let ntv = Cuda::new();
        let device = ntv.new_device(&ntv.hardwares()[0..1]).unwrap();
        let shared_data = &mut SharedTensor::<f32>::new(&device, &10).unwrap();
        match shared_data.get(&device) {
            Some(&MemoryType::Cuda(_)) => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn it_creates_new_shared_memory_for_opencl() {
        let ntv = OpenCL::new();
        let device = ntv.new_device(&ntv.hardwares()[0..1]).unwrap();
        let shared_data = &mut SharedTensor::<f32>::new(&device, &10).unwrap();
        match shared_data.get(&device) {
            Some(&MemoryType::OpenCL(_)) => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn it_syncs_from_native_to_cuda_and_back() {
        let cu = Cuda::new();
        let nt = Native::new();
        let cu_device = cu.new_device(&cu.hardwares()[0..1]).unwrap();
        let nt_device = nt.new_device(nt.hardwares()).unwrap();
        let mem = &mut SharedTensor::<f64>::new(&nt_device, &3).unwrap();
        write_to_memory(mem.get_mut(&nt_device).unwrap(), &[1, 2, 3]);
        mem.add_device(&cu_device).unwrap();
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
    #[cfg(feature = "opencl")]
    fn it_syncs_from_native_to_opencl_and_back() {
        let cl = OpenCL::new();
        let nt = Native::new();
        let cl_device = cl.new_device(&cl.hardwares()[0..1]).unwrap();
        let nt_device = nt.new_device(nt.hardwares()).unwrap();
        let mem = &mut SharedTensor::<f64>::new(&nt_device, &3).unwrap();
        write_to_memory(mem.get_mut(&nt_device).unwrap(), &[1, 2, 3]);
        mem.add_device(&cl_device).unwrap();
        match mem.sync(&cl_device) {
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
        let shared_data = &mut SharedTensor::<f32>::new(&cpu_dev, &10).unwrap();
        assert_eq!(&cpu_dev, shared_data.latest_device());
    }

    #[test]
    fn it_reshapes_correctly() {
        let ntv = Native::new();
        let cpu_dev = ntv.new_device(ntv.hardwares()).unwrap();
        let mut shared_data = &mut SharedTensor::<f32>::new(&cpu_dev, &10).unwrap();
        assert!(shared_data.reshape(&vec![5, 2]).is_ok());
    }

    #[test]
    fn it_returns_err_for_invalid_size_reshape() {
        let ntv = Native::new();
        let cpu_dev = ntv.new_device(ntv.hardwares()).unwrap();
        let mut shared_data = &mut SharedTensor::<f32>::new(&cpu_dev, &10).unwrap();
        assert!(shared_data.reshape(&vec![10, 2]).is_err());
    }
}
