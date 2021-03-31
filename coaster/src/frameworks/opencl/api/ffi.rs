//! Provides the Foreign Function Interface for OpenCL.
#![allow(non_camel_case_types, dead_code)]
#![allow(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations
)]

use super::types as cl;
use libc;

#[cfg_attr(target_os = "macos", link(name = "OpenCL", kind = "framework"))]
#[cfg_attr(not(target_os = "macos"), link(name = "OpenCL"))]
extern "C" {
    /* Platform APIs */
    pub fn clGetPlatformIDs(
        num_entries: cl::uint,
        platforms: *mut cl::platform_id,
        num_platforms: *mut cl::uint,
    ) -> cl::Status;

    pub fn clGetPlatformInfo(
        platform: cl::platform_id,
        param_name: cl::platform_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Device APIs */
    pub fn clGetDeviceIDs(
        platform: cl::platform_id,
        device_type: cl::device_type,
        num_entries: cl::uint,
        devices: *mut cl::device_id,
        num_devices: *mut cl::uint,
    ) -> cl::Status;

    pub fn clGetDeviceInfo(
        device: cl::device_id,
        param_name: cl::device_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Context APIs */
    pub fn clCreateContext(
        properties: *const cl::context_properties,
        num_devices: cl::uint,
        devices: *const cl::device_id,
        pfn_notify: extern "C" fn(
            *const libc::c_char,
            *const libc::c_void,
            libc::size_t,
            *mut libc::c_void,
        ),
        user_data: *mut libc::c_void,
        errcode_ret: *mut cl::int,
    ) -> cl::context_id;

    pub fn clCreateContextFromType(
        properties: *mut cl::context_properties,
        device_type: cl::device_type,
        pfn_notify: extern "C" fn(
            *mut libc::c_char,
            *mut libc::c_void,
            libc::size_t,
            *mut libc::c_void,
        ),
        user_data: *mut libc::c_void,
        errcode_ret: *mut cl::int,
    ) -> cl::context_id;

    pub fn clRetainContext(context: cl::context_id) -> cl::Status;

    pub fn clReleaseContext(context: cl::context_id) -> cl::Status;

    pub fn clGetContextInfo(
        context: cl::context_id,
        param_name: cl::context_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Command Queue APIs */
    pub fn clCreateCommandQueue(
        context: cl::context_id,
        device: cl::device_id,
        properties: cl::command_queue_properties,
        errcode_ret: *mut cl::int,
    ) -> cl::queue_id;

    pub fn clRetainCommandQueue(command_queue: cl::queue_id) -> cl::Status;

    pub fn clReleaseCommandQueue(command_queue: cl::queue_id) -> cl::Status;

    pub fn clGetCommandQueueInfo(
        command_queue: cl::queue_id,
        param_name: cl::command_queue_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Memory Object APIs */
    pub fn clCreateBuffer(
        context: cl::context_id,
        flags: cl::mem_flags,
        size: libc::size_t,
        host_ptr: *mut libc::c_void,
        errcode_ret: *mut cl::int,
    ) -> cl::memory_id;

    pub fn clCreateSubBuffer(
        buffer: cl::memory_id,
        flags: cl::mem_flags,
        buffer_create_type: cl::buffer_create_type,
        buffer_create_info: *mut libc::c_void,
        errcode_ret: *mut cl::int,
    ) -> cl::memory_id;

    pub fn clCreateImage2D(
        context: cl::context_id,
        flags: cl::mem_flags,
        image_format: *mut cl::image_format,
        image_width: libc::size_t,
        image_height: libc::size_t,
        image_row_pitch: libc::size_t,
        host_ptr: *mut libc::c_void,
        errcode_ret: *mut cl::int,
    ) -> cl::memory_id;

    pub fn clCreateImage3D(
        context: cl::context_id,
        flags: cl::mem_flags,
        image_format: *mut cl::image_format,
        image_width: libc::size_t,
        image_height: libc::size_t,
        image_depth: libc::size_t,
        image_row_pitch: libc::size_t,
        image_depth: libc::size_t,
        image_row_pitch: libc::size_t,
        image_slice_pitch: libc::size_t,
        host_ptr: *mut libc::c_void,
        errcode_ret: *mut cl::int,
    ) -> cl::memory_id;

    pub fn clRetainMemObject(memobj: cl::memory_id) -> cl::Status;

    pub fn clReleaseMemObject(memobj: cl::memory_id) -> cl::Status;

    pub fn clGetSupportedImageFormats(
        context: cl::context_id,
        flags: cl::mem_flags,
        image_type: cl::mem_object_type,
        num_entries: cl::uint,
        image_formats: *mut cl::image_format,
        num_image_formats: *mut cl::uint,
    ) -> cl::Status;

    pub fn clGetMemObjectInfo(
        memobj: cl::memory_id,
        param_name: cl::mem_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    pub fn clGetImageInfo(
        image: cl::memory_id,
        param_name: cl::image_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    pub fn clSetMemObjectDestructorCallback(
        memobj: cl::memory_id,
        pfn_notify: extern "C" fn(cl::memory_id, *mut libc::c_void),
        user_data: *mut libc::c_void,
    ) -> cl::Status;

    /*mut * Sampler APIs */
    pub fn clCreateSampler(
        context: cl::context_id,
        normalize_coords: cl::boolean,
        addressing_mode: cl::addressing_mode,
        filter_mode: cl::filter_mode,
        errcode_ret: *mut cl::int,
    ) -> cl::sampler;

    pub fn clRetainSampler(sampler: cl::sampler) -> cl::Status;

    pub fn clReleaseSampler(sampler: cl::sampler) -> cl::int;

    pub fn clGetSamplerInfo(
        sampler: cl::sampler,
        param_name: cl::sampler_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Program Object APIs */
    pub fn clCreateProgramWithSource(
        context: cl::context_id,
        count: cl::uint,
        strings: *const *const libc::c_char,
        lengths: *const libc::size_t,
        errcode_ret: *mut cl::int,
    ) -> cl::program;

    pub fn clCreateProgramWithBinary(
        context: cl::context_id,
        num_devices: cl::uint,
        device_list: *const cl::device_id,
        lengths: *const libc::size_t,
        binaries: *const *const libc::c_uchar,
        binary_status: *mut cl::int,
        errcode_ret: *mut cl::int,
    ) -> cl::program;

    pub fn clRetainProgram(program: cl::program) -> cl::Status;

    pub fn clReleaseProgram(program: cl::program) -> cl::Status;

    pub fn clBuildProgram(
        program: cl::program,
        num_devices: cl::uint,
        device_list: *const cl::device_id,
        options: *const libc::c_char,
        pfn_notify: extern "C" fn(cl::program, *mut libc::c_void),
        user_data: *mut libc::c_void,
    ) -> cl::Status;

    pub fn clUnloadCompiler() -> cl::Status;

    pub fn clGetProgramInfo(
        program: cl::program,
        param_name: cl::program_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    pub fn clGetProgramBuildInfo(
        program: cl::program,
        device: cl::device_id,
        param_name: cl::program_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Kernel Object APIs */
    pub fn clCreateKernel(
        program: cl::program,
        kernel_name: *const libc::c_char,
        errcode_ret: *mut cl::int,
    ) -> cl::kernel_id;

    pub fn clCreateKernelsInProgram(
        program: cl::program,
        num_kernels: cl::uint,
        kernels: *mut cl::kernel_id,
        num_kernels_ret: *mut cl::uint,
    ) -> cl::Status;

    pub fn clRetainKernel(kernel: cl::kernel_id) -> cl::Status;

    pub fn clReleaseKernel(kernel: cl::kernel_id) -> cl::Status;

    pub fn clSetKernelArg(
        kernel: cl::kernel_id,
        arg_index: cl::uint,
        arg_size: libc::size_t,
        arg_value: *const libc::c_void,
    ) -> cl::Status;

    pub fn clGetKernelInfo(
        kernel: cl::kernel_id,
        param_name: cl::kernel_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    pub fn clGetKernelWorkGroupInfo(
        kernel: cl::kernel_id,
        device: cl::device_id,
        param_name: cl::kernel_work_group_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Event Object APIs */
    pub fn clWaitForEvents(num_events: cl::uint, event_list: *const cl::event) -> cl::Status;

    pub fn clGetEventInfo(
        event: cl::event,
        param_name: cl::event_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    pub fn clCreateUserEvent(context: cl::context_id, errcode_ret: *mut cl::int) -> cl::event;

    pub fn clRetainEvent(event: cl::event) -> cl::Status;

    pub fn clReleaseEvent(event: cl::event) -> cl::Status;

    pub fn clSetUserEventStatus(event: cl::event, execution_status: cl::int) -> cl::Status;

    pub fn clSetEventCallback(
        event: cl::event,
        command_exec_callback_type: cl::int,
        pfn_notify: extern "C" fn(cl::event, cl::int, *mut libc::c_void),
        user_data: *mut libc::c_void,
    ) -> cl::Status;

    /* Profiling APIs */
    pub fn clGetEventProfilingInfo(
        event: cl::event,
        param_name: cl::profiling_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t,
    ) -> cl::Status;

    /* Flush and Finish APIs */
    pub fn clFlush(command_queue: cl::queue_id) -> cl::Status;

    pub fn clFinish(command_queue: cl::queue_id) -> cl::Status;

    /* Enqueued Commands APIs */
    pub fn clEnqueueReadBuffer(
        command_queue: cl::queue_id,
        buffer: cl::memory_id,
        blocking_read: cl::boolean,
        offset: libc::size_t,
        cb: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueReadBufferRect(
        command_queue: cl::queue_id,
        buffer: cl::memory_id,
        blocking_read: cl::boolean,
        buffer_origin: *mut libc::size_t,
        host_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        buffer_row_pitch: libc::size_t,
        buffer_slice_pitch: libc::size_t,
        host_row_pitch: libc::size_t,
        host_slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueWriteBuffer(
        command_queue: cl::queue_id,
        buffer: cl::memory_id,
        blocking_write: cl::boolean,
        offset: libc::size_t,
        cb: libc::size_t,
        ptr: *const libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueWriteBufferRect(
        command_queue: cl::queue_id,
        blocking_write: cl::boolean,
        buffer_origin: *mut libc::size_t,
        host_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        buffer_row_pitch: libc::size_t,
        buffer_slice_pitch: libc::size_t,
        host_row_pitch: libc::size_t,
        host_slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueCopyBuffer(
        command_queue: cl::queue_id,
        src_buffer: cl::memory_id,
        dst_buffer: cl::memory_id,
        src_offset: libc::size_t,
        dst_offset: libc::size_t,
        cb: libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueCopyBufferRect(
        command_queue: cl::queue_id,
        src_buffer: cl::memory_id,
        dst_buffer: cl::memory_id,
        src_origin: *mut libc::size_t,
        dst_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        src_row_pitch: libc::size_t,
        src_slice_pitch: libc::size_t,
        dst_row_pitch: libc::size_t,
        dst_slice_pitch: libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueReadImage(
        command_queue: cl::queue_id,
        image: cl::memory_id,
        blocking_read: cl::boolean,
        origin: *mut libc::size_t,
        region: *mut libc::size_t,
        row_pitch: libc::size_t,
        slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueWriteImage(
        command_queue: cl::queue_id,
        image: cl::memory_id,
        blocking_write: cl::boolean,
        origin: *mut libc::size_t,
        region: *mut libc::size_t,
        input_row_pitch: libc::size_t,
        input_slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueCopyImage(
        command_queue: cl::queue_id,
        src_image: cl::memory_id,
        dst_image: cl::memory_id,
        src_origin: *mut libc::size_t,
        dst_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueCopyImageToBuffer(
        command_queue: cl::queue_id,
        src_image: cl::memory_id,
        dst_buffer: cl::memory_id,
        src_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        dst_offset: libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueCopyBufferToImage(
        command_queue: cl::queue_id,
        src_buffer: cl::memory_id,
        dst_image: cl::memory_id,
        src_offset: libc::size_t,
        dst_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueMapBuffer(
        command_queue: cl::queue_id,
        buffer: cl::memory_id,
        blocking_map: cl::boolean,
        map_flags: cl::map_flags,
        offset: libc::size_t,
        cb: libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
        errorcode_ret: *mut cl::int,
    );

    pub fn clEnqueueMapImage(
        command_queue: cl::queue_id,
        image: cl::memory_id,
        blocking_map: cl::boolean,
        map_flags: cl::map_flags,
        origin: *mut libc::size_t,
        region: *mut libc::size_t,
        image_row_pitch: libc::size_t,
        image_slice_pitch: libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
        errorcode_ret: *mut cl::int,
    );

    pub fn clEnqueueUnmapMemObject(
        command_queue: cl::queue_id,
        memobj: cl::memory_id,
        mapped_ptr: *mut libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueNDRangeKernel(
        command_queue: cl::queue_id,
        kernel: cl::kernel_id,
        work_dim: cl::uint,
        global_work_offset: *const libc::size_t,
        global_work_size: *const libc::size_t,
        local_work_size: *const libc::size_t,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueTask(
        command_queue: cl::queue_id,
        kernel: cl::kernel_id,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueNativeKernel(
        command_queue: cl::queue_id,
        user_func: extern "C" fn(*mut libc::c_void),
        args: *mut libc::c_void,
        cb_args: libc::size_t,
        num_mem_objects: cl::uint,
        mem_list: *const cl::memory_id,
        args_mem_loc: *const *const libc::c_void,
        num_events_in_wait_list: cl::uint,
        event_wait_list: *const cl::event,
        event: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueMarker(command_queue: cl::queue_id, event: *mut cl::event) -> cl::Status;

    pub fn clEnqueueWaitForEvents(
        command_queue: cl::queue_id,
        num_events: cl::uint,
        event_list: *mut cl::event,
    ) -> cl::Status;

    pub fn clEnqueueBarrier(command_queue: cl::queue_id) -> cl::Status;

    /* Extension function access
     *
     * Returns the extension function address for the given function name,
     * or NULL if a valid function can not be found. The client must
     * check to make sure the address is not NULL, before using or
     * or calling the returned function address.
     */
    pub fn clGetExtensionFunctionAddress(func_name: *const libc::c_char) -> *mut libc::c_void;
}
