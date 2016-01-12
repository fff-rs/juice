extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod backend_spec {

    use co::backend::{Backend, BackendConfig};
    #[cfg(feature = "opencl")]
    use co::frameworks::OpenCL;
    use co::framework::IFramework;

    #[test]
    #[cfg(feature = "opencl")]
    fn it_works() {
        let framework = OpenCL::new();
        let hardwares = framework.hardwares().to_vec();
        let backend_config = BackendConfig::new(framework, &hardwares);
        let backend = Backend::new(backend_config);
        println!("{:?}", backend);
    }
}
