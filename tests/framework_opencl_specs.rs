extern crate coaster as co;
extern crate libc;

#[cfg(test)]
#[cfg(feature = "opencl")]
mod framework_opencl_spec {
    use co::prelude::*;
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
        let hardwares = &frm.hardwares()[0..1];
        println!("{:?}", frm.new_device(hardwares));
    }

    #[test]
    #[allow(unused_must_use)]
    fn it_creates_memory() {
        let frm = OpenCL::new();
        let ctx = frm.new_device(&frm.hardwares()[0..1]).unwrap();
        Memory::new(&ctx, 8);
    }

    #[test]
    fn it_creates_queue() {
        let frm = OpenCL::new();
        let ctx = frm.new_device(&frm.hardwares()[0..1]).unwrap();
        Queue::new(&ctx, &frm.hardwares()[0..1][0], None).unwrap();
    }
}
