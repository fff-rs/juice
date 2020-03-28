#![feature(test)]

extern crate coaster as co;
extern crate rcudnn;
extern crate test;






#[cfg(test)]
mod cudnn_spec {

    use rcudnn::Cudnn;

    #[test]
    #[serial]
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
    #[serial]
    fn it_returns_version() {
        println!("cuDNN Version: {:?}", Cudnn::version());
    }
}
