extern crate cudnn;

#[cfg(test)]
mod cudnn_spec {

    use cudnn::Cudnn;

    #[test]
    fn it_initializes_correctly() {
        /*
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
