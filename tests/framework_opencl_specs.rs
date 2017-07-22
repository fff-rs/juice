extern crate coaster as co;
extern crate libc;

#[cfg(test)]
#[cfg(feature = "opencl")]
mod framework_opencl_spec {
    use co::prelude::*;
    use co::frameworks::opencl::memory::*;
    use co::frameworks::opencl::queue::*;
    use co::frameworks::opencl::context::*;

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

    #[test]
    fn it_querries_context_info() {
        let frm = OpenCL::new();
        let ctx = frm.new_device(&frm.hardwares()[0..1]).unwrap();
        println!("ReferenceCount: {:?}", ctx.get_context_info(ContextInfoQuery::ReferenceCount));
        println!("NumDevices: {:?}", ctx.get_context_info(ContextInfoQuery::NumDevices));
        println!("Properties: {:?}", ctx.get_context_info(ContextInfoQuery::Properties));
        println!("Devices: {:?}", ctx.get_context_info(ContextInfoQuery::Devices));
	}
}
