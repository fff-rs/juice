//! Provides the generic functionality for a backend-specific implementation of a
//! [library][libraries].
//! [libraries]: ../libraries/index.html
//!
//! A binary defines one or (more often) many operations, which share related functionalities and are
//! provided by a specific [library][libraries] such as [Blas][blas].
//!
//! A binary needs to be 'built', which is handled by the specific framework implementation of a
//! binary representation, and returns initialized operations based on a [library][libraries].
//!
//! You are ususally not interacting with a binary itself, but rather use it to construct the
//! backend-agnostic operations, which can then be run and parallelized via an
//! unified interface - `backend.__name_of_the_operation__`.
//!
//! ## Development
//!
//! The funcionality provided here is used to construct specific Coaster binaries, which are
//! used to construct the basic computation behavior that come shipped with Coaster. However, it should
//! allow you to define and run your own backend-agnostic programs as well.
//!
//! [blas]: ../libraries/blas/index.html

/// Defines the functionality for turning a library into backend-specific, executable operations.
pub trait IBinary {
    // Returns the unique identifier of the Binary.
    //fn id(&self) -> isize;
    // Creates a HashMap of available, ready-to-use operations, based on the provided library and
    // tailored for a framework.
    //fn create_operations();
}
