#![feature(test)]

extern crate coaster as co;
extern crate rcudnn;
extern crate test;






#[cfg(test)]
mod cudnn_spec {

    use rcudnn::Cudnn;

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
