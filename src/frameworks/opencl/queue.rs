//! Provides a Rust wrapper around OpenCL's command queue.
//!
//! ## OpenCL Command Queue
//!
//! OpenCL objects such as memory, program and kernel objects are created using a context.
//! Operations on these objects are performed using a command-queue. The command-queue can be used
//! to queue a set of operations (referred to as commands) in order. Having multiple command-queues
//! allows applications to queue multiple independent commands without requiring synchronization.
//! Note that this should work as long as these objects are not being shared. Sharing of objects
//! across multiple command-queues will require the application to perform appropriate
//! synchronization.

use super::api::types as cl;

#[derive(Debug, Copy, Clone)]
/// Defines a OpenCL Queue.
pub struct Queue {
    id: isize,
}

impl Queue {
    /// Initializes a new OpenCL command queue.
    pub fn from_isize(id: isize) -> Queue {
        Queue { id: id }
    }

    /// Initializes a new OpenCL command queue from its C type.
    pub fn from_c(id: cl::queue_id) -> Queue {
        Queue { id: id as isize }
    }

    /// Returns the id as isize.
    pub fn id(&self) -> isize {
        self.id
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::queue_id {
        self.id as cl::queue_id
    }
}
