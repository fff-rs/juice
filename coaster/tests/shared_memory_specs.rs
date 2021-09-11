use coaster as co;

#[cfg(test)]
mod shared_memory_spec {
    use super::co::frameworks::native::flatbox::FlatBox;
    use super::co::prelude::*;
    use super::co::tensor::Error;

    #[cfg(features = "cuda")]
    fn write_to_memory<T: Copy>(mem: &mut FlatBox, data: &[T]) {
        let mem_buffer = mem.as_mut_slice::<T>();
        for (index, datum) in data.iter().enumerate() {
            mem_buffer[index] = *datum;
        }
    }

    #[test]
    #[cfg(feature = "native")]
    fn it_creates_new_shared_memory_for_native() {
        let ntv = Native::new();
        let cpu = ntv.new_device(ntv.hardwares()).unwrap();
        let mut shared_data = SharedTensor::<f32>::new(&10);
        let data = shared_data.write_only(&cpu).unwrap().as_slice::<f32>();
        assert_eq!(10, data.len());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn it_creates_new_shared_memory_for_cuda() {
        let ntv = Cuda::new();
        let device = ntv.new_device(&ntv.hardwares()[0..1]).unwrap();
        let mut shared_data = SharedTensor::<f32>::new(&10);
        shared_data.write_only(&device).unwrap();
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn it_creates_new_shared_memory_for_opencl() {
        let ntv = OpenCL::new();
        let device = ntv.new_device(&ntv.hardwares()[0..1]).unwrap();
        let mut shared_data = SharedTensor::<f32>::new(&10);
        shared_data.write_only(&device).unwrap();
    }

    #[test]
    #[cfg(feature = "native")]
    fn it_fails_on_initialized_memory_read() {
        let ntv = Native::new();
        let cpu = ntv.new_device(ntv.hardwares()).unwrap();
        let mut shared_data = SharedTensor::<f32>::new(&10);
        assert_eq!(
            shared_data.read(&cpu).unwrap_err(),
            Error::UninitializedMemory
        );
        assert_eq!(
            shared_data.read_write(&cpu).unwrap_err(),
            Error::UninitializedMemory
        );

        shared_data.write_only(&cpu).unwrap();
        shared_data.drop(&cpu).unwrap();

        assert_eq!(
            shared_data.read(&cpu).unwrap_err(),
            Error::UninitializedMemory
        );
    }

    #[test]
    #[cfg(features = "cuda")]
    fn it_syncs_from_native_to_cuda_and_back() {
        let cu = Cuda::new();
        let nt = Native::new();
        let cu_device = cu.new_device(&cu.hardwares()[0..1]).unwrap();
        let nt_device = nt.new_device(nt.hardwares()).unwrap();
        let mut mem = SharedTensor::<f64>::new(&3);
        write_to_memory(mem.write_only(&nt_device).unwrap(), &[1.0f64, 2.0, 123.456]);
        mem.read(&cu_device).unwrap();

        // It has successfully synced to the device.
        // Not the other way around.
        mem.drop(&nt_device).unwrap();
        assert_eq!(
            mem.read(&nt_device).unwrap().as_slice::<f64>(),
            [1.0, 2.0, 123.456]
        );
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn it_syncs_from_native_to_opencl_and_back() {
        let cl = OpenCL::new();
        let nt = Native::new();
        let cl_device = cl.new_device(&cl.hardwares()[0..1]).unwrap();
        let nt_device = nt.new_device(nt.hardwares()).unwrap();
        let mut mem = SharedTensor::<f64>::new(&3);
        write_to_memory(mem.write_only(&nt_device).unwrap(), &[1.0f64, 2.0, 123.456]);
        mem.read(&cl_device).unwrap();

        // It has not successfully synced to the device.
        // Not the other way around.
        mem.drop(&nt_device).unwrap();
        assert_eq!(
            mem.read(&nt_device).unwrap().as_slice::<f64>(),
            [1.0, 2.0, 123.456]
        );
    }

    #[test]
    fn it_reshapes_correctly() {
        let mut shared_data = SharedTensor::<f32>::new(&10);
        assert!(shared_data.reshape(&vec![5, 2]).is_ok());
    }

    #[test]
    fn it_returns_err_for_invalid_size_reshape() {
        let mut shared_data = SharedTensor::<f32>::new(&10);
        assert!(shared_data.reshape(&vec![10, 2]).is_err());
    }
}
