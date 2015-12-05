extern crate cudnn;

#[cfg(test)]
mod tensor_descriptor_spec {

    use cudnn::{TensorDescriptor, DataType};

    #[test]
    fn it_initializes_a_tensor_descriptor() {
        match TensorDescriptor::new(&[2, 2], DataType::Float) {
            Ok(tensor_desc) => println!("{:?}", tensor_desc.id_c()),
            Err(err) => {
                println!("{:?}", err);
                assert!(false);
            }
        }
    }
}
