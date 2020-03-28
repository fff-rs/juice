extern crate coaster as co;
extern crate libc;

#[cfg(test)]
mod backend_spec {
    #[cfg(feature = "native")]
    mod native {
        use std::rc::Rc;
        use crate::co::prelude::*;

        #[test]
        fn it_can_create_default_backend() {
            assert!(Backend::<Native>::default().is_ok());
        }

        #[test]
        fn it_can_use_ibackend_trait_object() {
            let framework = Native::new();
            let hardwares = framework.hardwares().to_vec();
            let backend_config = BackendConfig::new(framework, &hardwares);
            let backend = Rc::new(Backend::new(backend_config).unwrap());
            use_ibackend(backend);
        }

        fn use_ibackend<B: IBackend>(backend: Rc<B>) {
            let backend: Rc<dyn IBackend<F=B::F>> = backend.clone();
            backend.device();
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use crate::co::*;

        #[test]
        #[serial_test::serial]
        fn it_can_create_default_backend() {
            assert!(Backend::<Cuda>::default().is_ok());
        }
    }

    #[cfg(feature = "opencl")]
    mod opencl {
        use co::*;

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
