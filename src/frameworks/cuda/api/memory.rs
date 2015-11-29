//! Provides the Cuda API with its memory/buffer functionality.

use libc;
use super::{API, Error};
use frameworks::cuda::{Context, Memory};
use super::types as cl;
use super::ffi::*;

impl API {
    /// Allocates memory on the OpenCL device.
    ///
    /// A buffer object stores a one-dimensional collection of elements.  Elements of a buffer
    /// object can be a scalar data type (such as an int, float), vector data type, or a
    /// user-defined structure.
    /// Returns a memory id for the created buffer, which can now be writen to.
    pub fn create_buffer(context: Context) -> Result<cl::memory_id, Error> {
        unimplemented!()
    }

    /// Releases allocated memory from the OpenCL device.
    pub fn release_memory(memory: &mut Memory) -> Result<(), Error> {
        Ok(try!(unsafe {API::ffi_release_mem_object(memory.id_c())}))
    }

    /// Reads from a buffer to the host memory.
    ///
    /// With write_to_buffer you can do the opposite, write from the host memory to a buffer.
    pub fn read_from_buffer<T>(
        mem: Memory,
        blocking_read: cl::boolean,
        offset: libc::size_t,
        size: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> Result<(), Error> {
        unimplemented!();
    }

    unsafe fn ffi_create_buffer(
        context: cl::context_id,
        flags: cl::mem_flags,
        size: libc::size_t,
        host_ptr: *mut libc::c_void
    ) -> Result<cl::memory_id, Error> {
        unimplemented!()
    }

    unsafe fn ffi_release_mem_object(memobj: cl::memory_id) -> Result<(), Error> {
        unimplemented!()
    }

    unsafe fn ffi_enqueue_read_buffer(
        command_queue: cl::queue_id,
        buffer: cl::memory_id,
        blocking_read: cl::boolean,
        offset: libc::size_t,
        cb: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event
    ) -> Result<(), Error> {
        unimplemented!()
    }
}
