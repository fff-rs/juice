extern crate coaster as co;

#[cfg(test)]
#[cfg(feature = "native")]
mod framework_native_spec {
    use co::prelude::*;

    #[test]
    fn it_works() {
        let frm = Native::new();
        assert_eq!(frm.hardwares().len(), 1);
    }
}
