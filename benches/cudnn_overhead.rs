#![feature(test)]

extern crate test;
extern crate cudnn;
extern crate collenchyma as co;

use test::Bencher;
use co::backend::{Backend, BackendConfig};
use co::frameworks::Cuda;
use co::framework::IFramework;

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
