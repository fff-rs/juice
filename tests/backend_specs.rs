extern crate collenchyma as co;
extern crate libc;

#[cfg(test)]
mod backend_spec {


    #[cfg(feature = "native")]
    mod native {
        use co::backend::{IBackend, Backend};
        use co::frameworks::Native;

        #[test]
        fn it_can_create_default_backend() {
            assert!(Backend::<Native>::default().is_ok());
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use co::backend::{IBackend, Backend};
        use co::frameworks::Cuda;

        #[test]
        fn it_can_create_default_backend() {
            assert!(Backend::<Cuda>::default().is_ok());
        }
    }

    #[cfg(feature = "opencl")]
    mod opencl {
        use co::backend::{IBackend, Backend, BackendConfig};
        use co::frameworks::OpenCL;
        use co::framework::IFramework;

        #[test]
        fn it_can_create_default_backend() {
            assert!(Backend::<OpenCL>::default().is_ok());
        }

        #[test]
        fn it_can_manually_create_backend() {
            let framework = OpenCL::new();
            let hardwares = framework.hardwares().to_vec();
            let backend_config = BackendConfig::new(framework, &hardwares);
            let backend = Backend::new(backend_config);
            println!("{:?}", backend);
        }
    }
}
