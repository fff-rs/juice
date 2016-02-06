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
use super::api::{API, Error};
use super::Context;
use super::Device;

#[derive(Debug, Clone)]
#[allow(missing_copy_implementations)]
/// Defines a OpenCL Queue.
pub struct Queue {
    id: isize,
}

impl Queue {
    /// Create a new command queue for the provided `context` and `device`.
    ///
    /// If no `queue_flags` are provided, the defaults are used.
    pub fn new(context: &Context, device: &Device, queue_flags: Option<&QueueFlags>) -> Result<Queue, Error> {
        let default_flags = QueueFlags::default();
        let flags = queue_flags.unwrap_or(&default_flags);
        API::create_queue(context, device, flags)
    }

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

bitflags! {
    #[allow(missing_docs)]
    flags QueueFlags: cl::bitfield {
        #[allow(missing_docs)]
        const CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1 << 0,
        #[allow(missing_docs)]
        const CL_QUEUE_PROFILING_ENABLE              = 1 << 1,
    }
}

impl Default for QueueFlags {
    fn default() -> QueueFlags {
        QueueFlags::empty()
    }
}
