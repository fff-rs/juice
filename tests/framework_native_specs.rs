extern crate collenchyma as co;

#[cfg(test)]
mod framework_native_spec {

    use co::framework::IFramework;
    use co::frameworks::Native;

    #[test]
    fn it_works() {
        let frm = Native::new();
        assert_eq!(frm.hardwares().len(), 1);
    }
}
