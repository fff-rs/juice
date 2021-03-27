//! Provides informations about the software system, such as Cuda, CUDA, that contains the set of
//! components to support [hardwares][hardware] with kernel execution.
//! [hardware]: ../hardware/index.html
//!
//! ## Ontology
//!
//! Cuda device -> Hardware
//! Cuda context -> Device

extern {}

use crate::backend::{Backend, IBackend};
use crate::framework::IFramework;
pub use self::memory::Memory;
pub use self::context::Context;
pub use self::function::Function;
pub use self::module::Module;
pub use self::device::{Device, DeviceInfo};
pub use self::api::{Driver, DriverError};
use crate::cudnn::*;
use crate::cublas;
use crate::BackendConfig;

pub mod device;
pub mod context;
pub mod function;
pub mod memory;
pub mod module;
mod api;


/// Initialise the CUDA, CUBLAS, and CUDNN APIs
///
/// # Safety
/// CUDA, CUBLAS, and CUDNN are all initialised by external handles, and will be destroyed via
/// external API calls when the Cuda struct within the backend is destroyed.
pub fn get_cuda_backend() -> Backend<Cuda> {
    let framework = Cuda::new();
    let hardwares = framework.hardwares()[0..1].to_vec();
    let backend_config = BackendConfig::new(framework, &hardwares);
    let mut backend = Backend::new(backend_config).unwrap();
    // CUDA backend must be initialised before CUDA and CUDNN can be initialised.
    // Ordering of CUBLAS & CUDNN being initialised is unimportant.
    backend.framework.initialise_cublas().unwrap();
    backend.framework.initialise_cudnn().unwrap();
    backend
}

#[derive(Debug, Clone)]
/// Provides the Cuda Framework.
pub struct Cuda {
    hardwares: Vec<Device>,
    binary: Module,
    cudnn: Option<Cudnn>,
    cublas: Option<cublas::Context>,
}

impl Cuda {
    /// Create a handle to CUBLAS and assign it to CUDA Object
    ///
    /// Creating a handle when the CUDA object is created initially will cause CUDA_ERROR_LAUNCH_FAILED
    /// when an attempt is made to use the pointer. This can also affect global initialisation of
    /// the pointer, and so the initialise must run after the CUDA Driver is fully initialised, or
    /// (theoretically) a call is done to CUDA Free or DeviceSynchronise.
    pub fn initialise_cublas(&mut self) -> Result<(), crate::framework::Error> {
        self.cublas = {
            let mut context = cublas::Context::new().unwrap();
            context.set_pointer_mode(cublas::api::PointerMode::Device).unwrap();
            Some(context)
        };
        Ok(())
    }

    /// Create a handle to CUDNN and assign it to CUDA Object
    pub fn initialise_cudnn(&mut self) -> Result<(), crate::framework::Error> {
        self.cudnn = match Cudnn::new() {
            Ok(cudnn_ptr) => Some(cudnn_ptr),
            Err(_) => None
        };
        Ok(())
    }

    /// Return a reference to the CUDNN Handle
    pub fn cudnn(&self) -> &Cudnn {
        match &self.cudnn {
            Some(cudnn) => cudnn,
            None => panic!("Couldn't find a CUDNN Handle - Initialise CUDNN has not been called")
        }
    }

    /// Return a reference to the CUBLAS Handle
    pub fn cublas(&self) -> &cublas::Context {
        match &self.cublas {
            Some(cublas) => cublas,
            None => panic!("Couldn't find a CUBLAS Handle - Initialise CUBLAS has not been called")
        }
    }
}

impl IFramework for Cuda {
    type H = Device;
    type D = Context;
    type B = Module;

    fn ID() -> &'static str { "CUDA" }

    fn new() -> Cuda {
        // Init function must be called before any other function from the Cuda Driver API can be
        // called.
        if let Err(err) = Driver::init() {
            panic!("Unable to initialize Cuda Framework: {}", err);
        }
        match Cuda::load_hardwares() {
            Ok(hardwares) => {
                Cuda {
                    hardwares,
                    binary: Module::from_isize(1),
                    cudnn: None,
                    cublas: None,
                }
            },
            Err(err) => panic!("Could not initialize Cuda Framework, due to: {}", err)
        }
    }

    fn load_hardwares() -> Result<Vec<Device>, crate::framework::Error> {
        Ok(Driver::load_devices()?)
    }

    fn hardwares(&self) -> &[Device] {
        &self.hardwares
    }

    fn binary(&self) -> &Self::B {
        &self.binary
    }

    /// Creates a new Cuda context for computation.
    ///
    /// Cuda's context differs from OpenCL's context. Multi device support works different in Cuda.
    /// This function currently suppports only one device, but should be a wrapper for multi device support.
    fn new_device(&self, hardwares: &[Device]) -> Result<Self::D, crate::framework::Error> {
        let length = hardwares.len();
        match length {
            0 => Err(crate::framework::Error::Implementation("No device for context specified.".to_string())),
            1 => Ok(Context::new(hardwares[0].clone())?),
            _ => Err(crate::framework::Error::Implementation("Cuda's `new_device` method currently supports only one Harware for Device creation.".to_string()))
        }
    }
}

impl IBackend for Backend<Cuda> {
    type F = Cuda;

    fn device(&self) -> &Context {
        &self.device()
    }

    fn synchronize(&self) -> Result<(), crate::framework::Error> {
        Ok(self.device().synchronize()?)
    }
}
