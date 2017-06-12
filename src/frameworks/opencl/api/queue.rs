//! Provides the OpenCL API with its command queue functionality.
//!
//! At Coaster device can be understood as a synonym to OpenCL's context.

use frameworks::opencl::{API, Device, Error, Context, Queue, QueueFlags};
use super::types as cl;
use super::ffi::*;

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
}
