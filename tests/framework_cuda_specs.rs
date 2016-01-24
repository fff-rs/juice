extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod framework_cuda_spec {
    use co::framework::IFramework;
    use co::frameworks::Cuda;
    use co::device::DeviceType;
    use co::frameworks::cuda::memory::*;
    use co::tensor::*;

    #[test]
    fn it_works() {
        let frm = Cuda::new();
        println!("{:?}", frm.hardwares());
    }

    #[test]
    fn it_creates_context() {
        let frm = Cuda::new();
        let hardwares = &frm.hardwares()[0..1];
        println!("{:?}", frm.new_device(hardwares));
    }

    #[test]
    #[allow(unused_must_use)]
    fn it_allocates_memory() {
        let vec_a = vec![0isize, 1, 2, -3, 4, 5, 6, 7];
        let frm = Cuda::new();
        if let DeviceType::Cuda(_) = frm.new_device(&frm.hardwares()[0..1]).unwrap() {
            // Cuda memory
            Memory::new(vec_a.len());
        }
    }

    #[test]
    #[allow(unused_must_use)]
    // Create a lot of new CUDA devices, tests for correct dropping of device
    fn it_creates_a_lot_of_devices() {
        for _ in 0..256 {
            let cuda = Cuda::new();
            let _ = cuda.new_device(&cuda.hardwares()[0..1]).unwrap();
        }
    }

    #[test]
    #[allow(unused_must_use)]
    // Allocate 128mb blocks with dropping them in between, tests for correct freeing of memory
    fn it_allocates_4gb_memory_same_device() {
        let cuda = Cuda::new();
        let device = cuda.new_device(&cuda.hardwares()[0..1]).unwrap();
        for _ in 0..256 {
            let _ = &mut SharedTensor::<f32>::new(&device, &vec![256, 1024, 128]).unwrap();
        }
    }
}
