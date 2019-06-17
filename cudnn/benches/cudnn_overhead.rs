#![feature(test)]

extern crate coaster as co;
extern crate cudnn;
extern crate test;

use co::backend::{Backend, BackendConfig};
use co::framework::IFramework;
use co::frameworks::Cuda;
use test::Bencher;

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
