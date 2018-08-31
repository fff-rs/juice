//! Provides safe wrappers around various CUDA APIs.
//!
//! You can find wrappers for the<br/>
//! * CUDA Driver API
//! * CUDA cuDNN API

pub use self::driver::API as Driver;
pub use self::driver::ffi as DriverFFI;
pub use self::driver::Error as DriverError;

pub mod driver;
