extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod framework_cuda_spec {

    use co::framework::IFramework;
    use co::frameworks::Cuda;
    use co::device::DeviceType;
    use co::frameworks::cuda::memory::*;

    #[test]
    fn it_works() {
        let frm = Cuda::new();
        println!("{:?}", frm.hardwares());
    }

    //#[test]
    //fn it_creates_context() {
    //    let frm = OpenCL::new();
    //    let hardwares = frm.hardwares()[0..1].to_vec();
    //    println!("{:?}", frm.new_device(hardwares));
    //}

    // #[test]
    // #[allow(unused_must_use)]
    // fn it_allocates_memory() {
    //     let vec_a = vec![0isize, 1, 2, -3, 4, 5, 6, 7];
    //     let frm = OpenCL::new();
    //     if let DeviceType::OpenCL(ctx) = frm.new_device(frm.hardwares()[0..1].to_vec()).unwrap() {
    //         // OpenCL memory
    //         Memory::new(ctx.id_c(), vec_a.len());
    //     }
    // }
}
