extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod backend_spec {

    use co::backend::{Backend, BackendConfig};
    use co::frameworks::{OpenCL, Native};
    use co::framework::*;

    #[test]
    fn it_works() {
        let framework = OpenCL::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        let backend = Backend::new(backend_config);
        println!("{:?}", backend);
    }
}
