//! Provides the interface for running parallel computations on one ore many devices.
//!
//! This is the abstraction over which you are interacting with your devices. You can create a
//! backend for computation by first choosing a specifc [Framework][frameworks] such as OpenCL and
//! afterwards selecting one or many available hardwares to create a backend.
//!
//! A backend provides you with the functionality of managing the memory of the devices and copying
//! your objects from host to devices and the other way around. Additionally you can execute
//! operations in parallel through kernel functions on the device(s) of the backend.
//!
//! ## Architecture
//!
//! The initialization of a backend happens through the BackendConfig, which defines which
//! [framework][framework] should be used and which [programs][program] should be available for
//! parallel execution.
//!
//! [frameworks]: ../frameworks/index.html
//! [framework]: ../framework/index.html
//! [program]: ../program/index.html
//!
//! ## Examples
//!
//! ```
//! extern crate collenchyma as co;
//! use co::framework::*;
//! use co::backend::{Backend, BackendConfig};
//! use co::frameworks::OpenCL;
//! #[allow(unused_variables)]
//! fn main() {
//!     // Initialize a new Framewok.
//!     let framework = OpenCL::new();
//!     // After initialization, the available hardware through the Framework can be obtained.
//!     let hardwares = framework.hardwares();
//!     // Create a Backend configuration with
//!     // - a Framework and
//!     // - the available hardwares you would like to use for computation (turn into a device).
//!     let backend_config = BackendConfig::new(framework, hardwares);
//!     // Create a ready to go backend from the configuration.
//!     let backend = Backend::new(backend_config);
//! }
//! ```

use error::Error;
use framework::IFramework;
use frameworks::{Native, OpenCL, Cuda};
use device::{IDevice, DeviceType};
use libraries::blas::IBlas;
use libraries::blas as bl;
use shared_memory::SharedMemory;
use libraries::blas::{IOperationAsum, IOperationAxpy, IOperationCopy, IOperationDot,
                      IOperationNrm2, IOperationScale, IOperationSwap};

#[derive(Debug, Clone)]
/// Defines the main and highest struct of Collenchyma.
pub struct Backend<F: IFramework> {
    /// Provides the Framework.
    ///
    /// The Framework implementation such as OpenCL, CUDA, etc. defines, which should be used and
    /// determines which hardwares will be available and how parallel kernel functions can be
    /// executed.
    ///
    /// Default: [Native][native]
    ///
    /// [native]: ../frameworks/native/index.html
    framework: Box<F>,
    /// Provides a device, created from one or many hardwares, which are ready to execute kernel
    /// methods and synchronize memory.
    device: DeviceType,
}

/// Defines the functionality of the Backend.
impl<F: IFramework + Clone> Backend<F> {
    /// Initialize a new native Backend from a BackendConfig.
    pub fn new(config: BackendConfig<F>) -> Result<Backend<F>, Error> {
        let device = try!(config.framework.new_device(config.hardwares));
        Ok(
            Backend {
                framework: Box::new(config.framework),
                device: device,
            }
        )
    }

    /// Returns the available hardware.
    pub fn hardwares(&self) -> Vec<F::H> {
        self.framework.hardwares()
    }

    /// Returns the backend framework.
    pub fn framework(&self) -> &Box<F> {
        &self.framework
    }

    /// Returns the backend device.
    pub fn device(&self) -> &DeviceType {
        &self.device
    }

    /// Returns the blas binary.
    pub fn binary(&self) -> &F::B {
        self.framework().binary()
    }
}

/// Describes a Backend.
///
/// Serves as a marker trait and helps for extern implementation.
pub trait IBackend {
    /// Represents the Framework of a Backend.
    type F: IFramework + Clone;
}

impl IBackend for Backend<Native> {
    type F = Native;
}

impl IBackend for Backend<OpenCL> {
    type F = OpenCL;
}

impl IBackend for Backend<Cuda> {
    type F = Cuda;
}

impl IBlas<f32> for Backend<OpenCL> {
    type B = ::frameworks::opencl::Program;

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}

macro_rules! iblas_asum_for {
    ($t:ident, $b:ty) => (
        fn asum(&self, x: &mut SharedMemory<$t>, result: &mut SharedMemory<$t>) -> Result<(), ::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationAsum<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `result`"))),
                )
            ))
        }
    );
}

macro_rules! iblas_axpy_for {
    ($t:ident, $b:ty) => (
        fn axpy(&self, a: &mut SharedMemory<$t>, x: &mut SharedMemory<$t>, y: &mut SharedMemory<$t>) -> Result<(), ::error::Error> {
            match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            Ok(try!(
                <$b as IOperationAxpy<$t>>::compute(&self,
                    try!(a.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `a`"))),
                    try!(x.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `y`"))),
                )
            ))
        }
    );
}

macro_rules! iblas_copy_for {
    ($t:ident, $b:ty) => (
        fn copy(&self, x: &mut SharedMemory<$t>, y: &mut SharedMemory<$t>) -> Result<(), ::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationCopy<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `y`"))),
                )
            ))
        }
    );
}

macro_rules! iblas_dot_for {
    ($t:ident, $b:ty) => (
        fn dot(&self, x: &mut SharedMemory<$t>, y: &mut SharedMemory<$t>, result: &mut SharedMemory<$t>) -> Result<(), Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationDot<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `x`"))),
                    try!(y.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `y`"))),
                    try!(result.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `result`")))
                )
            ))
        }
    );
}

macro_rules! iblas_nrm2_for {
    ($t:ident, $b:ty) => (
        fn nrm2(&self, x: &mut SharedMemory<$t>, result: &mut SharedMemory<$t>) -> Result<(), ::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match result.add_device(self.device()) { _ => () }
            Ok(try!(
                <$b as IOperationNrm2<$t>>::compute(&self,
                    try!(x.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `x`"))),
                    try!(result.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `result`"))),
                )
            ))
        }
    );
}

macro_rules! iblas_scale_for {
    ($t:ident, $b:ty) => (
        fn scale(&self, a: &mut SharedMemory<$t>, x: &mut SharedMemory<$t>) -> Result<(), ::error::Error> {
            match a.add_device(self.device()) { _ => try!(a.sync(self.device())) }
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            Ok(try!(
                <$b as IOperationScale<$t>>::compute(&self,
                    try!(a.get(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `a`"))),
                    try!(x.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `x`"))),
                )
            ))
        }
    );
}

macro_rules! iblas_swap_for {
    ($t:ident, $b:ty) => (
        fn swap(&self, x: &mut SharedMemory<$t>, y: &mut SharedMemory<$t>) -> Result<(), ::error::Error> {
            match x.add_device(self.device()) { _ => try!(x.sync(self.device())) }
            match y.add_device(self.device()) { _ => try!(y.sync(self.device())) }
            Ok(try!(
                <$b as IOperationSwap<$t>>::compute(&self,
                    try!(x.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `x`"))),
                    try!(y.get_mut(self.device()).ok_or(bl::Error::MissingArgument("Unable to resolve memory for `y`"))),
                )
            ))
        }
    );
}

impl IBlas<f32> for Backend<Native> {
    type B = ::frameworks::native::Binary;

    iblas_asum_for!(f32, Backend<Native>);
    iblas_axpy_for!(f32, Backend<Native>);
    iblas_copy_for!(f32, Backend<Native>);
    iblas_dot_for!(f32, Backend<Native>);
    iblas_nrm2_for!(f32, Backend<Native>);
    iblas_scale_for!(f32, Backend<Native>);
    iblas_swap_for!(f32, Backend<Native>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}

impl IBlas<f64> for Backend<Native> {
    type B = ::frameworks::native::Binary;

    iblas_asum_for!(f64, Backend<Native>);
    iblas_axpy_for!(f64, Backend<Native>);
    iblas_copy_for!(f64, Backend<Native>);
    iblas_dot_for!(f64, Backend<Native>);
    iblas_nrm2_for!(f64, Backend<Native>);
    iblas_scale_for!(f64, Backend<Native>);
    iblas_swap_for!(f64, Backend<Native>);

    fn binary(&self) -> &Self::B {
        self.binary()
    }

    fn device(&self) -> &DeviceType {
        self.device()
    }
}

#[derive(Debug, Clone)]
/// Provides Backend Configuration.
///
/// Use it to initialize a new Backend.
pub struct BackendConfig<F: IFramework> {
    framework: F,
    hardwares: Vec<F::H>,
}

impl<F: IFramework + Clone> BackendConfig<F> {
    /// Creates a new BackendConfig.
    pub fn new(framework: F, hardwares: Vec<F::H>) -> BackendConfig<F> {
        BackendConfig {
            framework: framework.clone(),
            hardwares: hardwares,
        }
    }
}
