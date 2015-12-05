extern crate cudnn;
extern crate collenchyma as co;

#[cfg(test)]
mod cudnn_spec {

    use cudnn::Cudnn;
    use co::backend::{Backend, BackendConfig};
    use co::frameworks::Cuda;
    use co::framework::IFramework;

    #[test]
    fn it_initializes_correctly() {
        /*
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
        */
    }

    #[test]
    fn it_returns_version() {
        println!("cuDNN Version: {:?}", Cudnn::version());
    }
}
