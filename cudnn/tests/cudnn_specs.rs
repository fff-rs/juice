extern crate cudnn;
extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod cudnn_spec {

    use cudnn::{Cudnn, DataType, TensorDescriptor, ScalParams};
    use co::backend::{Backend, BackendConfig};
    use co::frameworks::Cuda;
    use co::framework::IFramework;
    use std::mem::transmute;

    #[test]
    fn it_initializes_correctly() {
        let cuda = Cuda::new();
        println!("{:?}", cuda.hardwares());
        cuda.new_device(cuda.hardwares()[0..1].to_vec());
        match Cudnn::new() {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("{:?}", err);
                assert!(false);
            }
        }
    }

    #[test]
    fn it_returns_version() {
        println!("cuDNN Version: {:?}", Cudnn::version());
    }

    #[test]
    fn it_computes_sigmoid() {
        let cuda = Cudnn::new().unwrap();
        let desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        match cuda.sigmoid_forward(
            &desc, unsafe { transmute::<u64, *const ::libc::c_void>(1u64) }, &desc, unsafe { transmute::<u64, *mut ::libc::c_void>(1u64) }, ScalParams::<f32>::default()
        ) {
            Ok(_) => assert!(true),
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }
}
