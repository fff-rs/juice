//! Provides a Rust wrapper around OpenCL's events.
//!
//! ## OpenCL Event
//!
//! Most OpenCL operations happen asynchronously on the OpenCL Device.
//! To provide the possibility to order and synchronize multiple operations,
//! the execution of an operation yields a Event object.
//! This Event can be used as input to other operations
//! which will wait until this Event has finished executing to run.

use super::api::types as cl;

#[derive(Debug, Copy, Clone)]
/// Defines a OpenCL Event;
pub struct Event {
    id: isize,
}

impl Event {
    /// Initializes a new OpenCL even from its C type.
    pub fn from_c(id: cl::event) -> Event {
        Event { id: id as isize }
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::event {
        self.id as cl::event
    }
}
