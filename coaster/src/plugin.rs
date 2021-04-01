//! Provides helpers for explicit implementations of Backend [Operations][operation].
//! [operation]: ../operation/index.html
//!
//! A Backend is a Rust struct like any other, therefore you probably would like to implement
//! certain methods for the Backend. As the whole purpose of a Backend is to provide an
//! abstraction over various computation devices and computation languages, these implemented
//! methods will than be able to excute on different devices and use the full power of the machine's
//! underlying hardware.
//!
//! So extending the Backend with operations is easy. In Coaster we call crates, which provide
//! operations for the Backend, Plugins. Plugins are usually a group of related operations of a common
//! field. Two examples for Coaster Plugins are [BLAS][coaster-blas] and [NN][coaster-nn].
//!
//! A Plugin does roughly two important things. It provides generic traits and the explicit implementation
//! of these traits for one or (even better) all available Coaster Frameworks - common host CPU, OpenCL,
//! CUDA.
//!
//! The structure of Plugin is pretty simple with as little overhead as possible. Macros make implementations
//! even easier. If you would like to use specific Plugins for you Backend, all you need to do is
//! set them as dependencies in your Cargo file in addition to the Coaster crate. The Plugin
//! then automatically extends the Backend provided by Coaster.
//!
//! Extending the Backend with your own Plugin is a straight forward process.
//! For now we recommend that you take a look at the general code structure of [Coaster-BLAS][coaster-blas]
//! or its documentation. Let us now about your Plugin on the Gitter chat, we are happy to feature
//! your Coaster Plugin on the README.
//!
//! [program]: ../program/index.html
//! [coaster-blas]: https://github.com/spearow/coaster-blas
//! [coaster-nn]: https://github.com/spearow/coaster-nn

pub use self::numeric_helpers::Float;
use crate::tensor;

/// Describes numeric types and traits for a Plugin.
pub mod numeric_helpers {
    pub use num::traits::*;
}

#[derive(Debug, thiserror::Error)]
/// Defines a high-level Plugin Error.
pub enum Error {
    /// Failure related to `SharedTensor`: use of uninitialized memory,
    /// synchronization error or memory allocation failure.
    #[error("SharedTensor error")]
    SharedTensor(#[from] tensor::Error),
    /// Failure at the execution of the Operation.
    #[error("Operation error")]
    Operation(&'static str),

    #[error("Plugin error: {0}")]
    Plugin(&'static str),

    /// Failure at the Plugin with an inner error type.
    #[error(transparent)]
    PluginInner(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
}
