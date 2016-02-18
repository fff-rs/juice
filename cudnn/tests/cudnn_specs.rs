extern crate cudnn;
extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod cudnn_spec {

    use cudnn::{Cudnn, API, TensorDescriptor, FilterDescriptor, ConvolutionDescriptor};
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
    */

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
