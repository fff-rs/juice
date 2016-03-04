//! Provides the generic functionality for backend-agnostic operations.
//!
//! An Operation describes the execution of a [library][library] provided functionality for a
//! specific [framework][frameworks]. An Operation can easily be executed in parallel on multi-core
//! devices. An Operation is a very similar to a usual function and defines usually one or many
//! arguments over which the operation then will happen.
//!
//! You are usually not interacting with an operation directly. To execute an operation you would
//! use the [backend][backend]. Also you will rarely initialize your operations directly,
//! as this happens automatically at the initialization of a [binary][binary].
//!
//! ## Development
//!
//! The functionality provided by this module is used to construct the basic operations that come
//! shipped with Collenchyma, but should also allow you to define and run your own backend-agnostic
//! operations as well.
//!
//! [frameworks]: ../frameworks/index.html
//! [backend]: ../backend/index.html
//! [library]: ../library/index.html
//! [binary]: ../binary/index.html

/// Defines the functionality of an operation.
pub trait IOperation {}
