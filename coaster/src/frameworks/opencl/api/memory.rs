//! Provides the OpenCL API with its memory/buffer functionality.
//!
//! At Coaster device can be understood as a synonym to OpenCL's context.

use super::ffi::*;
use super::types as cl;
use frameworks::opencl::{Context, Error, Event, Memory, MemoryFlags, Queue, API};
use libc;
use std::ptr;

impl API {
    /// Allocates memory on the OpenCL device.
    ///
    /// A buffer object stores a one-dimensional collection of elements.  Elements of a buffer
    /// object can be a scalar data type (such as an int, float), vector data type, or a
    /// user-defined structure.
    /// Returns a memory id for the created buffer, which can now be writen to.
    pub fn create_buffer(
        context: &Context,
        flags: MemoryFlags,
        size: usize,
        host_pointer: Option<*mut u8>,
    ) -> Result<Memory, Error> {
        let host_ptr = host_pointer.unwrap_or(ptr::null_mut());
        Ok(Memory::from_c(unsafe {
            API::ffi_create_buffer(
                context.id() as *mut libc::c_void,
                flags.bits(),
                size,
                host_ptr as *mut libc::c_void,
            )
        }?))
    }

    /// Releases allocated memory from the OpenCL device.
    pub fn release_memory(memory: &mut Memory) -> Result<(), Error> {
        Ok(unsafe { API::ffi_release_mem_object(memory.id_c()) }?)
    }

    /// Reads from a OpenCL memory object to the host memory.
    ///
    /// With write_to_memory you can do the opposite, write from the host memory to a OpenCL memory object.
    pub fn read_from_memory(
        queue: &Queue,
        mem: &Memory,
        blocking_read: bool,
        offset: usize,
        size: usize,
        host_mem: *mut libc::c_void,
        event_wait_list: &[Event],
    ) -> Result<Event, Error> {
        let num_events_in_wait_list = event_wait_list.len();
        let event_list: *const *mut libc::c_void = if !event_wait_list.is_empty() {
            event_wait_list.as_ptr() as *const *mut libc::c_void
        } else {
            ptr::null_mut()
        };
        let new_event: cl::event = 0 as *mut libc::c_void;
        let res = unsafe {
            API::ffi_enqueue_read_buffer(
                queue.id_c(),
                mem.id_c(),
                blocking_read as cl::boolean,
                offset,
                size,
                host_mem,
                num_events_in_wait_list as cl::uint,
                event_list,
                new_event as *mut cl::event,
            )
        };
        match res {
            Ok(_) => Ok(Event::from_c(new_event)),
            Err(err) => Err(err),
        }
    }

    /// Write to a OpenCL memory object from host memory.
    ///
    /// With read_from_memory you can do the opposite, read from a OpenCL memory object to host memory.
    pub fn write_to_memory(
        queue: &Queue,
        mem: &mut Memory,
        blocking_write: bool,
        offset: usize,
        size: usize,
        host_mem: *const libc::c_void,
        event_wait_list: &[Event],
    ) -> Result<Event, Error> {
        let num_events_in_wait_list = event_wait_list.len();
        let event_list: *const *mut libc::c_void = if !event_wait_list.is_empty() {
            event_wait_list.as_ptr() as *const *mut libc::c_void
        } else {
            ptr::null_mut()
        };
        let new_event: cl::event = 0 as *mut libc::c_void;
        let res = unsafe {
            API::ffi_enqueue_write_buffer(
                queue.id_c(),
                mem.id_c(),
                blocking_write as cl::boolean,
                offset,
                size,
                host_mem,
                num_events_in_wait_list as cl::uint,
                event_list,
                new_event as *mut cl::event,
            )
        };
        match res {
            Ok(_) => Ok(Event::from_c(new_event)),
            Err(err) => Err(err),
        }
    }

    unsafe fn ffi_create_buffer(
        context: cl::context_id,
        flags: cl::mem_flags,
        size: libc::size_t,
        host_ptr: *mut libc::c_void,
    ) -> Result<cl::memory_id, Error> {
        let mut errcode: i32 = 0;
        let memory_id = clCreateBuffer(context, flags, size, host_ptr, &mut errcode);
        match errcode {
            errcode if errcode == cl::Status::SUCCESS as i32 => Ok(memory_id),
            errcode if errcode == cl::Status::INVALID_CONTEXT as i32 => Err(Error::InvalidContext("context is not a valid context.")),
            errcode if errcode == cl::Status::INVALID_VALUE as i32 => Err(Error::InvalidValue("values specified in flags are not valid")),
            errcode if errcode == cl::Status::INVALID_BUFFER_SIZE as i32 => Err(Error::InvalidBufferSize("size is 0^10")),
            errcode if errcode == cl::Status::INVALID_HOST_PTR as i32 => Err(Error::InvalidHostPtr("if host_ptr is NULL and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags or if host_ptr is not NULL but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags")),
            errcode if errcode == cl::Status::MEM_OBJECT_ALLOCATION_FAILURE as i32 => Err(Error::MemObjectAllocationFailure("failure toallocate memory for buffer object.")),
            errcode if errcode == cl::Status::OUT_OF_RESOURCES as i32 => Err(Error::OutOfResources("Failure to allocate resources on the device")),
            errcode if errcode == cl::Status::OUT_OF_HOST_MEMORY as i32 => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to create memory buffer."))
        }
    }

    unsafe fn ffi_release_mem_object(memobj: cl::memory_id) -> Result<(), Error> {
        match clReleaseMemObject(memobj) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_MEM_OBJECT => Err(Error::InvalidMemObject(
                "memobj is not a valid memory object.",
            )),
            cl::Status::OUT_OF_RESOURCES => Err(Error::OutOfResources(
                "Failure to allocate resources on the device",
            )),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory(
                "Failure to allocate resources on the host",
            )),
            _ => Err(Error::Other("Unable to release memory object.")),
        }
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
        event: *mut cl::event,
    ) -> Result<(), Error> {
        match clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_COMMAND_QUEUE => Err(Error::InvalidCommandQueue("command_queue is not a valid command-queue")),
            cl::Status::INVALID_CONTEXT => Err(Error::InvalidContext("the context associated with command_queue and buffer are not the same or if the context associated with command_queue and events in event_wait_list are not the same.")),
            cl::Status::INVALID_MEM_OBJECT => Err(Error::InvalidMemObject("buffer is not a valid memory object.")),
            cl::Status::INVALID_VALUE => Err(Error::InvalidValue("the region being read or written specified by (offset, size) is out of bounds or if ptris a NULLvalueor if sizeis 0.")),
            cl::Status::INVALID_EVENT_WAIT_LIST => Err(Error::InvalidEventWaitList("event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_listis not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events.")),
            cl::Status::MISALIGNED_SUB_BUFFER_OFFSET => Err(Error::MisalignedSubBufferOffset("buffer is a sub-buffer object and offsetspecified when the sub-buffer object is created is not aligned to CL_DEVICE_MEM_BASE_ADDR_ALIGN value for device associated with queue.")),
            cl::Status::EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST => Err(Error::ExecStatusErrorForEventsInWaitList("the read and write operations are blocking and the execution status of any of the events in event_wait_listis a negative integer value.")),
            cl::Status::MEM_OBJECT_ALLOCATION_FAILURE => Err(Error::MemObjectAllocationFailure("there is a failure to allocate memory fordata store associated with buffer.")),
            cl::Status::INVALID_OPERATION => Err(Error::InvalidOperation("called on buffer which has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.")),
            cl::Status::OUT_OF_RESOURCES => Err(Error::OutOfResources("Failure to allocate resources on the device")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to enqueue read buffer."))
        }
    }

    unsafe fn ffi_enqueue_write_buffer(
        command_queue: cl::queue_id,
        buffer: cl::memory_id,
        blocking_write: cl::boolean,
        offset: libc::size_t,
        cb: libc::size_t,
        ptr: *const libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> Result<(), Error> {
        match clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_COMMAND_QUEUE => Err(Error::InvalidCommandQueue("command_queue is not a valid command-queue")),
            cl::Status::INVALID_CONTEXT => Err(Error::InvalidContext("the context associated with command_queue and buffer are not the same or if the context associated with command_queue and events in event_wait_list are not the same.")),
            cl::Status::INVALID_MEM_OBJECT => Err(Error::InvalidMemObject("buffer is not a valid memory object.")),
            cl::Status::INVALID_VALUE => Err(Error::InvalidValue("the region being read or written specified by (offset, size) is out of bounds or if ptris a NULLvalueor if sizeis 0.")),
            cl::Status::INVALID_EVENT_WAIT_LIST => Err(Error::InvalidEventWaitList("event_wait_list is NULL and num_events_in_wait_list > 0, or event_wait_listis not NULL and num_events_in_wait_list is 0, or if event objects in event_wait_list are not valid events.")),
            cl::Status::MEM_OBJECT_ALLOCATION_FAILURE => Err(Error::MemObjectAllocationFailure("there is a failure to allocate memory fordata store associated with buffer.")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to enqueue write buffer."))
        }
    }
}
