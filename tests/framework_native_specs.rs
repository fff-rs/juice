extern crate collenchyma as co;

#[cfg(test)]
#[cfg(feature = "native")]
mod framework_native_spec {
    use co::*;

    #[test]
    fn it_works() {
        let frm = Native::new();
        assert_eq!(frm.hardwares().len(), 1);
    }
}
