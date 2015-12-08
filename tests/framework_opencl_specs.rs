extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod framework_opencl_spec {

    use co::device::DeviceType;
    use co::framework::IFramework;
    use co::frameworks::OpenCL;
    use co::frameworks::opencl::memory::*;
    use co::frameworks::opencl::queue::*;

    #[test]
    fn it_works() {
        let frm = OpenCL::new();
        println!("{:?}", frm.hardwares());
    }

    #[test]
    fn it_creates_context() {
        let frm = OpenCL::new();
        let hardwares = frm.hardwares()[0..1].to_vec();
        println!("{:?}", frm.new_device(hardwares));
    }

    #[test]
    #[allow(unused_must_use)]
    fn it_creates_memory() {
        let frm = OpenCL::new();
        if let DeviceType::OpenCL(ref ctx) = frm.new_device(frm.hardwares()[0..1].to_vec()).unwrap() {
            Memory::new(ctx, 8);
        }
    }

    #[test]
    fn it_creates_queue() {
        let frm = OpenCL::new();
        if let DeviceType::OpenCL(ref ctx) = frm.new_device(frm.hardwares()[0..1].to_vec()).unwrap() {
            assert!(Queue::new(ctx, &frm.hardwares()[0..1][0], None).is_ok());
        }
    }
}
