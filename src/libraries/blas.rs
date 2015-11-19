//! Provides backend-agnostic BLAS operations.

use backend::Backend;
use hardware::IHardware;
use device::IDevice;
use framework::IFramework;
use frameworks::Native;
use shared_memory::SharedMemory;
use blas::Vector;

#[derive(Debug, Copy, Clone)]
/// Blas
pub struct Blas;

/// Addition
pub trait Plus {
    /// plus
    fn plus<T>(self, a: SharedMemory<T>, b: SharedMemory<T>, c: &mut SharedMemory<T>);
}

impl<F: IFramework + Plus + Clone> Plus for Backend<F> {
    fn plus<T>(self, a: SharedMemory<T>, b: SharedMemory<T>, c: &mut SharedMemory<T>) {
        self.framework().plus(a, b, c)
    }
}
