extern crate rcudnn as cudnn;

#[cfg(test)]
mod tensor_descriptor_spec {

    use crate::cudnn::utils::DataType;
    use crate::cudnn::TensorDescriptor;

    #[test]
    fn it_initializes_a_tensor_descriptor() {
        match TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float) {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("{:?}", err);
                assert!(false);
            }
        }
    }
}
