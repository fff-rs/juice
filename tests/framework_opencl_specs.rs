extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod framework_opencl_spec {

    use co::framework::IFramework;
    use co::frameworks::OpenCL;
    use co::frameworks::opencl::memory::*;

    use co::memory::*;

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
    fn it_allocates_memory() {
        // let (device, ctx, queue) = create_compute_context().unwrap();

        //let vec_a = vec![0isize, 1, 2, -3, 4, 5, 6, 7];
        //let frm = OpenCL::new();
        //let ctx = frm.new_device(frm.hardwares()[0..1].to_vec()).unwrap();
        // OpenCL memory
        // let ctx_ptr = ctx.ctx as *mut libc::c_void;
        //let res = Memory::<Vec<isize>>::new(ctx.id_c(), vec_a.len());
        // pinned host memory
        //let bx = Box::new(vec_a.clone());
        //let res = Memory::<Vec<isize>>::from_box(ctx.id_c(), bx);
    }

    #[test]
    fn it_creates_buffer() {
        // let vec_a = vec![0isize, 1, 2, -3, 4, 5, 6, 7];
        // let mut buf = Buffer::new();
        //
        // let frm = OpenCL::new();
        // let dev = frm.new_device(frm.hardwares()[0..1].to_vec()).unwrap();
        // let mem = &mut dev.alloc_memory::<isize>(vec_a.len());
        //
        // buf.add_copy(&dev, mem);
    }
}
