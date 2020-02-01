extern crate cudnn;
extern crate collenchyma as co;
extern crate libc;

extern crate cudnn_sys as ffi;
use ffi::*;

#[cfg(test)]
mod cudnn_spec {

    use cudnn::{Cudnn, API, TensorDescriptor, ActivationDescriptor, FilterDescriptor, ConvolutionDescriptor};
    use cudnn::utils::DataType;
    use co::frameworks::Cuda;
    use co::framework::IFramework;

    #[test]
    fn it_initializes_correctly() {
        let cuda = Cuda::new();
        println!("{:?}", cuda.hardwares());
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

    /*
     * Results sometimes in weird memory allocation problems, with other tests if we run this test.
     * Might be due to the strange and totally not actually working memory pointers
     * `unsafe { transmute::<u64, *const ::libc::c_void>(1u64) }`.
     *
     * Since then this has been rewritten to not use transmute but a sequence of unsafe optimizations.
     */
    #[test]
    fn it_computes_sigmoid() {
        let cudnn = Cudnn::new().unwrap();
        let desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        let acti = ActivationDescriptor::new(::cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID).unwrap();

        let mut a : u64 = 1;
        let a_ptr : *mut u64 = &mut a;
        let mut b : u64 = 1;
        let b_ptr : *mut u64 = &mut b;
        unsafe {
            let mut x : *mut ::libc::c_void = ::std::ptr::null_mut();;
            ::cudaHostAlloc(&mut x, 2*2*2, 0);
            match API::activation_forward(
                *cudnn.id_c(),
                *acti.id_c(),
                a_ptr as *mut ::libc::c_void,
                *desc.id_c(),
                x,
                b_ptr as *mut ::libc::c_void,
                *desc.id_c(),
                x,
            ) {
                Ok(_) => assert!(true),
                Err(err) => { println!("{:?}", err); assert!(false) }
            }
            ::cudaFreeHost(x);
        }
    }


    #[test]
    fn it_finds_correct_convolution_algorithm_forward() {
        let cudnn = Cudnn::new().unwrap();
        let src = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        let filter = FilterDescriptor::new(&[1, 1, 1], DataType::Float).unwrap();
        let conv = ConvolutionDescriptor::new(&[1, 1, 1], &[1, 1, 1], DataType::Float).unwrap();
        let dest = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        match API::find_convolution_forward_algorithm(*cudnn.id_c(), *filter.id_c(), *conv.id_c(), *src.id_c(), *dest.id_c()) {
            Ok(algos) => { assert_eq!(2, algos.len())},
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }

    #[test]
    fn it_finds_correct_convolution_algorithm_backward() {
        let cudnn = Cudnn::new().unwrap();
        let src = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        let filter = FilterDescriptor::new(&[1, 1, 1], DataType::Float).unwrap();
        let conv = ConvolutionDescriptor::new(&[1, 1, 1], &[1, 1, 1], DataType::Float).unwrap();
        let dest = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        match API::find_convolution_backward_data_algorithm(*cudnn.id_c(), *filter.id_c(), *conv.id_c(), *src.id_c(), *dest.id_c()) {
            Ok(algos) => { assert_eq!(2, algos.len())},
            Err(err) => { println!("{:?}", err); assert!(false) }
        }
    }
}
