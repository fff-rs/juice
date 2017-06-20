//! Provides the OpenCL API with its command queue functionality.
//!
//! At Coaster device can be understood as a synonym to OpenCL's context.

use frameworks::opencl::{API, Device, Error, Context, Queue, QueueFlags};
use super::types as cl;
use super::ffi::*;
use libc;

impl API {
    /// Returns a command queue for a specified context and device.
    ///
    /// OpenCL command queues are used to control memory allocation and operations
    /// for a single device.
    pub fn create_queue(context: &Context, device: &Device, queue_flags: &QueueFlags) -> Result<Queue, Error> {
        Ok(Queue::from_c(try!(unsafe {
            API::ffi_create_command_queue(context.id_c(), device.id_c(), queue_flags.bits())
        })))
    }

    /// Releases command queue from the OpenCL device.
    pub fn release_queue(queue: &mut Queue) -> Result<(), Error> {
        Ok(try!(unsafe {API::ffi_release_command_queue(queue.id_c())}))
    }

    unsafe fn ffi_create_command_queue(
        context: cl::context_id,
        device: cl::device_id,
        properties: cl::bitfield,
    ) -> Result<cl::queue_id, Error> {
        let mut errcode: i32 = 0;
        let queue_id = clCreateCommandQueue(context, device, properties, &mut errcode);
        match errcode {
            errcode if errcode == cl::Status::SUCCESS as i32 => Ok(queue_id),
            errcode if errcode == cl::Status::INVALID_CONTEXT as i32 => Err(Error::InvalidContext("context is not a valid context.")),
            errcode if errcode == cl::Status::INVALID_DEVICE as i32 => Err(Error::InvalidDevice("devices contains an invalid device.")),
            errcode if errcode == cl::Status::INVALID_VALUE as i32 => Err(Error::InvalidValue("values specified in flags are not valid")),
            errcode if errcode == cl::Status::INVALID_QUEUE_PROPERTIES as i32 => Err(Error::InvalidQueueProperties("values specified in properties are valid but are not supported by the device")),
            errcode if errcode == cl::Status::OUT_OF_HOST_MEMORY as i32 => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to create command queue."))
        }
    }

    unsafe fn ffi_release_command_queue(command_queue: cl::queue_id) -> Result<(), Error> {
        match clReleaseCommandQueue(command_queue) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_COMMAND_QUEUE => Err(Error::InvalidCommandQueue("command_queue is not a valid command-queue")),
            _ => Err(Error::Other("Unable to release command queue."))
        }
    }

    unsafe fn ffi_enqueue_nd_range_kernel (
        command_queue: cl::queue_id,
        kernel: cl::kernel_id,
        work_dim: cl::uint,
        global_work_offset: *const libc::size_t,
        global_work_size: *const libc::size_t,
        local_work_size: *const libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event
        ) -> Result<(),Error> {
        match clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event) {
            cl::Status::SUCCESS => Ok(()),
            cl::Status::INVALID_COMMAND_QUEUE => Err(Error::InvalidCommandQueue("command_queue is not a valid command-queue")),
            cl::Status::INVALID_CONTEXT => Err(Error::InvalidContext("the context associated with command_queue and buffer are not the same or if the context associated with command_queue and events in event_wait_list are not the same.")),
            cl::Status::INVALID_KERNEL => Err(Error::InvalidKernel("kernel is not a valid kernel object")),
            cl::Status::INVALID_KERNEL_ARGS => Err(Error::InvalidKernelArgs("if the kernel argument values have not been specified")),
            cl::Status::INVALID_WORK_DIMENSION => Err(Error::InvalidWorkDimension("work_dim is not a valid value")),
            cl::Status::INVALID_WORK_GROUP_SIZE => Err(Error::InvalidWorkGroupSize("xx1")),
            cl::Status::INVALID_WORK_ITEM_SIZE => Err(Error::InvalidWorkItemSize("xx2")),
            cl::Status::INVALID_GLOBAL_OFFSET => Err(Error::InvalidGlobalOffset("xx3")),
            cl::Status::INVALID_EVENT_WAIT_LIST => Err(Error::InvalidEventWaitList("xx4")),
            cl::Status::MEM_OBJECT_ALLOCATION_FAILURE => Err(Error::MemObjectAllocationFailure("there is a failure to allocate memory fordata store associated with buffer.")),
            cl::Status::INVALID_OPERATION => Err(Error::InvalidOperation("called on buffer which has been created with CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS.")),
            cl::Status::OUT_OF_RESOURCES => Err(Error::OutOfResources("Failure to allocate resources on the device")),
            cl::Status::OUT_OF_HOST_MEMORY => Err(Error::OutOfHostMemory("Failure to allocate resources on the host")),
            _ => Err(Error::Other("Unable to enqueue read buffer."))
        }
    }

    pub fn enqueue_kernel() {
        // TODO ffi_enqueue_nd_range_kernel
    }
}
