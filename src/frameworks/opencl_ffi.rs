#![allow(non_camel_case_types, dead_code)]
#![allow(missing_docs, missing_debug_implementations, missing_copy_implementations)]

use libc;
use std::fmt;
use num::FromPrimitive;
use framework::FrameworkError;

/* Opaque types */
pub type cl_platform_id                 = *mut libc::c_void;
pub type cl_device_id                   = *mut libc::c_void;
pub type cl_context                     = *mut libc::c_void;
pub type cl_command_queue               = *mut libc::c_void;
pub type cl_mem                         = *mut libc::c_void;
pub type cl_program                     = *mut libc::c_void;
pub type cl_kernel                      = *mut libc::c_void;
pub type cl_event                       = *mut libc::c_void;
pub type cl_sampler                     = *mut libc::c_void;

/* Scalar types */
pub type cl_char                        = i8;
pub type cl_uchar                       = u8;
pub type cl_short                       = i16;
pub type cl_ushort                      = u16;
pub type cl_int                         = i32;
pub type cl_uint                        = u32;
pub type cl_long                        = i64;
pub type cl_ulong                       = u64;
pub type cl_half                        = u16;
pub type cl_float                       = f32;
pub type cl_double                      = f64;

pub type cl_bool                        = cl_uint;
pub type cl_bitfield                    = cl_ulong;
pub type cl_device_type                 = cl_bitfield;
pub type cl_platform_info               = cl_uint;
pub type cl_device_info                 = cl_uint;
pub type cl_device_fp_config            = cl_bitfield;
pub type cl_device_mem_cache_type       = cl_uint;
pub type cl_device_local_mem_type       = cl_uint;
pub type cl_device_exec_capabilities    = cl_bitfield;
pub type cl_command_queue_properties    = cl_bitfield;

pub type cl_context_properties          = libc::intptr_t;
pub type cl_context_info                = cl_uint;
pub type cl_command_queue_info          = cl_uint;
pub type cl_channel_order               = cl_uint;
pub type cl_channel_type                = cl_uint;
pub type cl_mem_flags                   = cl_bitfield;
pub type cl_mem_object_type             = cl_uint;
pub type cl_mem_info                    = cl_uint;
pub type cl_image_info                  = cl_uint;
pub type cl_buffer_create_type          = cl_uint;
pub type cl_addressing_mode             = cl_uint;
pub type cl_filter_mode                 = cl_uint;
pub type cl_sampler_info                = cl_uint;
pub type cl_map_flags                   = cl_bitfield;
pub type cl_program_info                = cl_uint;
pub type cl_program_build_info          = cl_uint;
pub type cl_build_status                = cl_int;
pub type cl_kernel_info                 = cl_uint;
pub type cl_kernel_work_group_info      = cl_uint;
pub type cl_event_info                  = cl_uint;
pub type cl_command_type                = cl_uint;
pub type cl_profiling_info              = cl_uint;

pub struct cl_image_format {
    image_channel_order:        cl_channel_order,
    image_channel_data_type:    cl_channel_type
}

pub struct cl_buffer_region {
    origin:     libc::size_t,
    size:       libc::size_t
}



enum_from_primitive! {
/// OpenCL error codes.
#[derive(PartialEq, Debug)]
#[repr()]
pub enum CLStatus {
    CL_SUCCESS = 0,
    CL_DEVICE_NOT_FOUND = -1,
    CL_DEVICE_NOT_AVAILABLE = -2,
    CL_COMPILER_NOT_AVAILABLE = -3,
    CL_MEM_OBJECT_ALLOCATION_FAILURE = -4,
    CL_OUT_OF_RESOURCES = -5,
    CL_OUT_OF_HOST_MEMORY = -6,
    CL_PROFILING_INFO_NOT_AVAILABLE = -7,
    CL_MEM_COPY_OVERLAP = -8,
    CL_IMAGE_FORMAT_MISMATCH = -9,
    CL_IMAGE_FORMAT_NOT_SUPPORTED = -10,
    CL_BUILD_PROGRAM_FAILURE = -11,
    CL_MAP_FAILURE = -12,
    CL_MISALIGNED_SUB_BUFFER_OFFSET = -13,
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14,
    CL_INVALID_VALUE = -30,
    CL_INVALID_DEVICE_TYPE = -31,
    CL_INVALID_PLATFORM = -32,
    CL_INVALID_DEVICE = -33,
    CL_INVALID_CONTEXT = -34,
    CL_INVALID_QUEUE_PROPERTIES = -35,
    CL_INVALID_COMMAND_QUEUE = -36,
    CL_INVALID_HOST_PTR = -37,
    CL_INVALID_MEM_OBJECT = -38,
    CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39,
    CL_INVALID_IMAGE_SIZE = -40,
    CL_INVALID_SAMPLER = -41,
    CL_INVALID_BINARY = -42,
    CL_INVALID_BUILD_OPTIONS = -43,
    CL_INVALID_PROGRAM = -44,
    CL_INVALID_PROGRAM_EXECUTABLE = -45,
    CL_INVALID_KERNEL_NAME = -46,
    CL_INVALID_KERNEL_DEFINITION = -47,
    CL_INVALID_KERNEL = -48,
    CL_INVALID_ARG_INDEX = -49,
    CL_INVALID_ARG_VALUE = -50,
    CL_INVALID_ARG_SIZE = -51,
    CL_INVALID_KERNEL_ARGS = -52,
    CL_INVALID_WORK_DIMENSION = -53,
    CL_INVALID_WORK_GROUP_SIZE = -54,
    CL_INVALID_WORK_ITEM_SIZE = -55,
    CL_INVALID_GLOBAL_OFFSET = -56,
    CL_INVALID_EVENT_WAIT_LIST = -57,
    CL_INVALID_EVENT = -58,
    CL_INVALID_OPERATION = -59,
    CL_INVALID_GL_OBJECT = -60,
    CL_INVALID_BUFFER_SIZE = -61,
    CL_INVALID_MIP_LEVEL = -62,
    CL_INVALID_GLOBAL_WORK_SIZE = -63,
    CL_INVALID_PROPERTY = -64,
    CL_PLATFORM_NOT_FOUND_KHR = -1001,
}
}

impl fmt::Display for CLStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/* OpenCL Version */
pub static CL_VERSION_1_0:                               cl_bool = 1;
pub static CL_VERSION_1_1:                               cl_bool = 1;

/* cl_bool */
pub static CL_FALSE:                                     cl_bool = 0;
pub static CL_TRUE:                                      cl_bool = 1;

/* cl_platform_info */
pub static CL_PLATFORM_PROFILE:                          cl_uint = 0x0900;
pub static CL_PLATFORM_VERSION:                          cl_uint = 0x0901;
pub static CL_PLATFORM_NAME:                             cl_uint = 0x0902;
pub static CL_PLATFORM_VENDOR:                           cl_uint = 0x0903;
pub static CL_PLATFORM_EXTENSIONS:                       cl_uint = 0x0904;

/* cl_device_type - bitfield */
pub static CL_DEVICE_TYPE_DEFAULT:                       cl_bitfield = 1 << 0;
pub static CL_DEVICE_TYPE_CPU:                           cl_bitfield = 1 << 1;
pub static CL_DEVICE_TYPE_GPU:                           cl_bitfield = 1 << 2;
pub static CL_DEVICE_TYPE_ACCELERATOR:                   cl_bitfield = 1 << 3;
pub static CL_DEVICE_TYPE_ALL:                           cl_bitfield = 0xFFFFFFFF;

/* cl_device_info */
pub static CL_DEVICE_TYPE:                               cl_uint = 0x1000;
pub static CL_DEVICE_VENDOR_ID:                          cl_uint = 0x1001;
pub static CL_DEVICE_MAX_COMPUTE_UNITS:                  cl_uint = 0x1002;
pub static CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:           cl_uint = 0x1003;
pub static CL_DEVICE_MAX_WORK_GROUP_SIZE:                cl_uint = 0x1004;
pub static CL_DEVICE_MAX_WORK_ITEM_SIZES:                cl_uint = 0x1005;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:        cl_uint = 0x1006;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:       cl_uint = 0x1007;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:         cl_uint = 0x1008;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:        cl_uint = 0x1009;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:       cl_uint = 0x100A;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:      cl_uint = 0x100B;
pub static CL_DEVICE_MAX_CLOCK_FREQUENCY:                cl_uint = 0x100C;
pub static CL_DEVICE_ADDRESS_BITS:                       cl_uint = 0x100D;
pub static CL_DEVICE_MAX_READ_IMAGE_ARGS:                cl_uint = 0x100E;
pub static CL_DEVICE_MAX_WRITE_IMAGE_ARGS:               cl_uint = 0x100F;
pub static CL_DEVICE_MAX_MEM_ALLOC_SIZE:                 cl_uint = 0x1010;
pub static CL_DEVICE_IMAGE2D_MAX_WIDTH:                  cl_uint = 0x1011;
pub static CL_DEVICE_IMAGE2D_MAX_HEIGHT:                 cl_uint = 0x1012;
pub static CL_DEVICE_IMAGE3D_MAX_WIDTH:                  cl_uint = 0x1013;
pub static CL_DEVICE_IMAGE3D_MAX_HEIGHT:                 cl_uint = 0x1014;
pub static CL_DEVICE_IMAGE3D_MAX_DEPTH:                  cl_uint = 0x1015;
pub static CL_DEVICE_IMAGE_SUPPORT:                      cl_uint = 0x1016;
pub static CL_DEVICE_MAX_PARAMETER_SIZE:                 cl_uint = 0x1017;
pub static CL_DEVICE_MAX_SAMPLERS:                       cl_uint = 0x1018;
pub static CL_DEVICE_MEM_BASE_ADDR_ALIGN:                cl_uint = 0x1019;
pub static CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:           cl_uint = 0x101A;
pub static CL_DEVICE_SINGLE_FP_CONFIG:                   cl_uint = 0x101B;
pub static CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:              cl_uint = 0x101C;
pub static CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:          cl_uint = 0x101D;
pub static CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:              cl_uint = 0x101E;
pub static CL_DEVICE_GLOBAL_MEM_SIZE:                    cl_uint = 0x101F;
pub static CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:           cl_uint = 0x1020;
pub static CL_DEVICE_MAX_CONSTANT_ARGS:                  cl_uint = 0x1021;
pub static CL_DEVICE_LOCAL_MEM_TYPE:                     cl_uint = 0x1022;
pub static CL_DEVICE_LOCAL_MEM_SIZE:                     cl_uint = 0x1023;
pub static CL_DEVICE_ERROR_CORRECTION_SUPPORT:           cl_uint = 0x1024;
pub static CL_DEVICE_PROFILING_TIMER_RESOLUTION:         cl_uint = 0x1025;
pub static CL_DEVICE_ENDIAN_LITTLE:                      cl_uint = 0x1026;
pub static CL_DEVICE_AVAILABLE:                          cl_uint = 0x1027;
pub static CL_DEVICE_COMPILER_AVAILABLE:                 cl_uint = 0x1028;
pub static CL_DEVICE_EXECUTION_CAPABILITIES:             cl_uint = 0x1029;
pub static CL_DEVICE_QUEUE_PROPERTIES:                   cl_uint = 0x102A;
pub static CL_DEVICE_NAME:                               cl_uint = 0x102B;
pub static CL_DEVICE_VENDOR:                             cl_uint = 0x102C;
pub static CL_DRIVER_VERSION:                            cl_uint = 0x102D;
pub static CL_DEVICE_PROFILE:                            cl_uint = 0x102E;
pub static CL_DEVICE_VERSION:                            cl_uint = 0x102F;
pub static CL_DEVICE_EXTENSIONS:                         cl_uint = 0x1030;
pub static CL_DEVICE_PLATFORM:                           cl_uint = 0x1031;
/* 0x1032 reserved for CL_DEVICE_DOUBLE_FP_CONFIG */
/* 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG */
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:        cl_uint = 0x1034;
pub static CL_DEVICE_HOST_UNIFIED_MEMORY:                cl_uint = 0x1035;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:           cl_uint = 0x1036;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:          cl_uint = 0x1037;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:            cl_uint = 0x1038;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:           cl_uint = 0x1039;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:          cl_uint = 0x103A;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:         cl_uint = 0x103B;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:           cl_uint = 0x103C;
pub static CL_DEVICE_OPENCL_C_VERSION:                   cl_uint = 0x103D;

/* cl_device_fp_config - bitfield */
pub static CL_FP_DENORM:                                 cl_bitfield = 1 << 0;
pub static CL_FP_INF_NAN:                                cl_bitfield = 1 << 1;
pub static CL_FP_ROUND_TO_NEAREST:                       cl_bitfield = 1 << 2;
pub static CL_FP_ROUND_TO_ZERO:                          cl_bitfield = 1 << 3;
pub static CL_FP_ROUND_TO_INF:                           cl_bitfield = 1 << 4;
pub static CL_FP_FMA:                                    cl_bitfield = 1 << 5;
pub static CL_FP_SOFT_FLOAT:                             cl_bitfield = 1 << 6;

/* cl_device_mem_cache_type */
pub static CL_NONE:                                      cl_uint = 0x0;
pub static CL_READ_ONLY_CACHE:                           cl_uint = 0x1;
pub static CL_READ_WRITE_CACHE:                          cl_uint = 0x2;

/* cl_device_local_mem_type */
pub static CL_LOCAL:                                     cl_uint = 0x1;
pub static CL_GLOBAL:                                    cl_uint = 0x2;

/* cl_device_exec_capabilities - bitfield */
pub static CL_EXEC_KERNEL:                               cl_bitfield = 1 << 0;
pub static CL_EXEC_NATIVE_KERNEL:                        cl_bitfield = 1 << 1;

/* cl_command_queue_properties - bitfield */
pub static CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE:       cl_bitfield = 1 << 0;
pub static CL_QUEUE_PROFILING_ENABLE:                    cl_bitfield = 1 << 1;

/* cl_context_info  */
pub static CL_CONTEXT_REFERENCE_COUNT:                   cl_uint = 0x1080;
pub static CL_CONTEXT_DEVICES:                           cl_uint = 0x1081;
pub static CL_CONTEXT_PROPERTIES:                        cl_uint = 0x1082;
pub static CL_CONTEXT_NUM_DEVICES:                       cl_uint = 0x1083;

/* cl_context_info + cl_context_properties */
pub static CL_CONTEXT_PLATFORM:                          libc::intptr_t = 0x1084;

/* cl_command_queue_info */
pub static CL_QUEUE_CONTEXT:                             cl_uint = 0x1090;
pub static CL_QUEUE_DEVICE:                              cl_uint = 0x1091;
pub static CL_QUEUE_REFERENCE_COUNT:                     cl_uint = 0x1092;
pub static CL_QUEUE_PROPERTIES:                          cl_uint = 0x1093;

/* cl_mem_flags - bitfield */
pub static CL_MEM_READ_WRITE:                            cl_bitfield = 1 << 0;
pub static CL_MEM_WRITE_ONLY:                            cl_bitfield = 1 << 1;
pub static CL_MEM_READ_ONLY:                             cl_bitfield = 1 << 2;
pub static CL_MEM_USE_HOST_PTR:                          cl_bitfield = 1 << 3;
pub static CL_MEM_ALLOC_HOST_PTR:                        cl_bitfield = 1 << 4;
pub static CL_MEM_COPY_HOST_PTR:                         cl_bitfield = 1 << 5;

/* cl_channel_order */
pub static CL_R:                                         cl_uint = 0x10B0;
pub static CL_A:                                         cl_uint = 0x10B1;
pub static CL_RG:                                        cl_uint = 0x10B2;
pub static CL_RA:                                        cl_uint = 0x10B3;
pub static CL_RGB:                                       cl_uint = 0x10B4;
pub static CL_RGBA:                                      cl_uint = 0x10B5;
pub static CL_BGRA:                                      cl_uint = 0x10B6;
pub static CL_ARGB:                                      cl_uint = 0x10B7;
pub static CL_INTENSITY:                                 cl_uint = 0x10B8;
pub static CL_LUMINANCE:                                 cl_uint = 0x10B9;
pub static CL_Rx:                                        cl_uint = 0x10BA;
pub static CL_RGx:                                       cl_uint = 0x10BB;
pub static CL_RGBx:                                      cl_uint = 0x10BC;

/* cl_channel_type */
pub static CL_SNORM_INT8:                                cl_uint = 0x10D0;
pub static CL_SNORM_INT16:                               cl_uint = 0x10D1;
pub static CL_UNORM_INT8:                                cl_uint = 0x10D2;
pub static CL_UNORM_INT16:                               cl_uint = 0x10D3;
pub static CL_UNORM_SHORT_565:                           cl_uint = 0x10D4;
pub static CL_UNORM_SHORT_555:                           cl_uint = 0x10D5;
pub static CL_UNORM_INT_101010:                          cl_uint = 0x10D6;
pub static CL_SIGNED_INT8:                               cl_uint = 0x10D7;
pub static CL_SIGNED_INT16:                              cl_uint = 0x10D8;
pub static CL_SIGNED_INT32:                              cl_uint = 0x10D9;
pub static CL_UNSIGNED_INT8:                             cl_uint = 0x10DA;
pub static CL_UNSIGNED_INT16:                            cl_uint = 0x10DB;
pub static CL_UNSIGNED_INT32:                            cl_uint = 0x10DC;
pub static CL_HALF_FLOAT:                                cl_uint = 0x10DD;
pub static CL_FLOAT:                                     cl_uint = 0x10DE;

/* cl_mem_object_type */
pub static CL_MEM_OBJECT_BUFFER:                         cl_uint = 0x10F0;
pub static CL_MEM_OBJECT_IMAGE2D:                        cl_uint = 0x10F1;
pub static CL_MEM_OBJECT_IMAGE3D:                        cl_uint = 0x10F2;

/* cl_mem_info */
pub static CL_MEM_TYPE:                                  cl_uint = 0x1100;
pub static CL_MEM_FLAGS:                                 cl_uint = 0x1101;
pub static CL_MEM_SIZE:                                  cl_uint = 0x1102;
pub static CL_MEM_HOST_PTR:                              cl_uint = 0x1103;
pub static CL_MEM_MAP_COUNT:                             cl_uint = 0x1104;
pub static CL_MEM_REFERENCE_COUNT:                       cl_uint = 0x1105;
pub static CL_MEM_CONTEXT:                               cl_uint = 0x1106;
pub static CL_MEM_ASSOCIATED_MEMOBJECT:                  cl_uint = 0x1107;
pub static CL_MEM_OFFSET:                                cl_uint = 0x1108;

/* cl_image_info */
pub static CL_IMAGE_FORMAT:                              cl_uint = 0x1110;
pub static CL_IMAGE_ELEMENT_SIZE:                        cl_uint = 0x1111;
pub static CL_IMAGE_ROW_PITCH:                           cl_uint = 0x1112;
pub static CL_IMAGE_SLICE_PITCH:                         cl_uint = 0x1113;
pub static CL_IMAGE_WIDTH:                               cl_uint = 0x1114;
pub static CL_IMAGE_HEIGHT:                              cl_uint = 0x1115;
pub static CL_IMAGE_DEPTH:                               cl_uint = 0x1116;

/* cl_addressing_mode */
pub static CL_ADDRESS_NONE:                              cl_uint = 0x1130;
pub static CL_ADDRESS_CLAMP_TO_EDGE:                     cl_uint = 0x1131;
pub static CL_ADDRESS_CLAMP:                             cl_uint = 0x1132;
pub static CL_ADDRESS_REPEAT:                            cl_uint = 0x1133;
pub static CL_ADDRESS_MIRRORED_REPEAT:                   cl_uint = 0x1134;

/* cl_filter_mode */
pub static CL_FILTER_NEAREST:                            cl_uint = 0x1140;
pub static CL_FILTER_LINEAR:                             cl_uint = 0x1141;

/* cl_sampler_info */
pub static CL_SAMPLER_REFERENCE_COUNT:                   cl_uint = 0x1150;
pub static CL_SAMPLER_CONTEXT:                           cl_uint = 0x1151;
pub static CL_SAMPLER_NORMALIZED_COORDS:                 cl_uint = 0x1152;
pub static CL_SAMPLER_ADDRESSING_MODE:                   cl_uint = 0x1153;
pub static CL_SAMPLER_FILTER_MODE:                       cl_uint = 0x1154;

/* cl_map_flags - bitfield */
pub static CL_MAP_READ:                                  cl_bitfield = 1 << 0;
pub static CL_MAP_WRITE:                                 cl_bitfield = 1 << 1;

/* cl_program_info */
pub static CL_PROGRAM_REFERENCE_COUNT:                   cl_uint = 0x1160;
pub static CL_PROGRAM_CONTEXT:                           cl_uint = 0x1161;
pub static CL_PROGRAM_NUM_DEVICES:                       cl_uint = 0x1162;
pub static CL_PROGRAM_DEVICES:                           cl_uint = 0x1163;
pub static CL_PROGRAM_SOURCE:                            cl_uint = 0x1164;
pub static CL_PROGRAM_BINARY_SIZES:                      cl_uint = 0x1165;
pub static CL_PROGRAM_BINARIES:                          cl_uint = 0x1166;

/* cl_program_build_info */
pub static CL_PROGRAM_BUILD_STATUS:                      cl_uint = 0x1181;
pub static CL_PROGRAM_BUILD_OPTIONS:                     cl_uint = 0x1182;
pub static CL_PROGRAM_BUILD_LOG:                         cl_uint = 0x1183;

/* cl_build_status */
pub static CL_BUILD_SUCCESS:                             cl_uint = 0;
pub static CL_BUILD_NONE:                                cl_uint = (-1isize) as cl_uint;
pub static CL_BUILD_ERROR:                               cl_uint = -2isize as cl_uint;
pub static CL_BUILD_IN_PROGRESS:                         cl_uint = -3isize as cl_uint;

/* cl_kernel_info */
pub static CL_KERNEL_FUNCTION_NAME:                      cl_uint = 0x1190;
pub static CL_KERNEL_NUM_ARGS:                           cl_uint = 0x1191;
pub static CL_KERNEL_REFERENCE_COUNT:                    cl_uint = 0x1192;
pub static CL_KERNEL_CONTEXT:                            cl_uint = 0x1193;
pub static CL_KERNEL_PROGRAM:                            cl_uint = 0x1194;

/* cl_kernel_work_group_info */
pub static CL_KERNEL_WORK_GROUP_SIZE:                    cl_uint = 0x11B0;
pub static CL_KERNEL_COMPILE_WORK_GROUP_SIZE:            cl_uint = 0x11B1;
pub static CL_KERNEL_LOCAL_MEM_SIZE:                     cl_uint = 0x11B2;
pub static CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: cl_uint = 0x11B3;
pub static CL_KERNEL_PRIVATE_MEM_SIZE:                   cl_uint = 0x11B4;

/* cl_event_info  */
pub static CL_EVENT_COMMAND_QUEUE:                       cl_uint = 0x11D0;
pub static CL_EVENT_COMMAND_TYPE:                        cl_uint = 0x11D1;
pub static CL_EVENT_REFERENCE_COUNT:                     cl_uint = 0x11D2;
pub static CL_EVENT_COMMAND_EXECUTION_STATUS:            cl_uint = 0x11D3;
pub static CL_EVENT_CONTEXT:                             cl_uint = 0x11D4;

/* cl_command_type */
pub static CL_COMMAND_NDRANGE_KERNEL:                    cl_uint = 0x11F0;
pub static CL_COMMAND_TASK:                              cl_uint = 0x11F1;
pub static CL_COMMAND_NATIVE_KERNEL:                     cl_uint = 0x11F2;
pub static CL_COMMAND_READ_BUFFER:                       cl_uint = 0x11F3;
pub static CL_COMMAND_WRITE_BUFFER:                      cl_uint = 0x11F4;
pub static CL_COMMAND_COPY_BUFFER:                       cl_uint = 0x11F5;
pub static CL_COMMAND_READ_IMAGE:                        cl_uint = 0x11F6;
pub static CL_COMMAND_WRITE_IMAGE:                       cl_uint = 0x11F7;
pub static CL_COMMAND_COPY_IMAGE:                        cl_uint = 0x11F8;
pub static CL_COMMAND_COPY_IMAGE_TO_BUFFER:              cl_uint = 0x11F9;
pub static CL_COMMAND_COPY_BUFFER_TO_IMAGE:              cl_uint = 0x11FA;
pub static CL_COMMAND_MAP_BUFFER:                        cl_uint = 0x11FB;
pub static CL_COMMAND_MAP_IMAGE:                         cl_uint = 0x11FC;
pub static CL_COMMAND_UNMAP_MEM_OBJECT:                  cl_uint = 0x11FD;
pub static CL_COMMAND_MARKER:                            cl_uint = 0x11FE;
pub static CL_COMMAND_ACQUIRE_GL_OBJECTS:                cl_uint = 0x11FF;
pub static CL_COMMAND_RELEASE_GL_OBJECTS:                cl_uint = 0x1200;
pub static CL_COMMAND_READ_BUFFER_RECT:                  cl_uint = 0x1201;
pub static CL_COMMAND_WRITE_BUFFER_RECT:                 cl_uint = 0x1202;
pub static CL_COMMAND_COPY_BUFFER_RECT:                  cl_uint = 0x1203;
pub static CL_COMMAND_USER:                              cl_uint = 0x1204;

/* command execution status */
pub static CL_COMPLETE:                                  cl_uint = 0x0;
pub static CL_RUNNING:                                   cl_uint = 0x1;
pub static CL_SUBMITTED:                                 cl_uint = 0x2;
pub static CL_QUEUED:                                    cl_uint = 0x3;

/* cl_buffer_create_type  */
pub static CL_BUFFER_CREATE_TYPE_REGION:                 cl_uint = 0x1220;

/* cl_profiling_info  */
pub static CL_PROFILING_COMMAND_QUEUED:                  cl_uint = 0x1280;
pub static CL_PROFILING_COMMAND_SUBMIT:                  cl_uint = 0x1281;
pub static CL_PROFILING_COMMAND_START:                   cl_uint = 0x1282;
pub static CL_PROFILING_COMMAND_END:                     cl_uint = 0x1283;

extern
{
    /* Platform APIs */
    /// Obtain a list of available platforms.
    ///
    /// *num_entries* is the number of cl_platform_id entries that can be added to platforms.
    /// If platforms is not NULL, the num_entries must be greater than zero.
    ///
    /// *platforms* returns a list of OpenCL platforms found. The cl_platform_id values returned in
    /// platforms can be used to identify a specific OpenCL platform. If platforms argument is
    /// NULL this argument is ignored. The number of OpenCL platforms returned is the minimum of
    /// the value specified by num_entries or the number of OpenCL platforms available.
    ///
    /// *num_platforms* returns the number of OpenCL platforms available. If num_platforms is NULL,
    /// this argument is ignored.
    pub fn clGetPlatformIDs(
        num_entries: cl_uint,
        platforms: *mut cl_platform_id,
        num_platforms: *mut cl_uint) -> cl_int;

    pub fn clGetPlatformInfo(
        platform: cl_platform_id,
        param_name: cl_platform_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;


    /* Device APIs */
    pub fn clGetDeviceIDs(
        platform: cl_platform_id,
        device_type: cl_device_type,
        num_entries: cl_uint,
        devices: *mut cl_device_id,
        num_devices: *mut cl_uint) -> cl_int;

    pub fn clGetDeviceInfo(
        device: cl_device_id,
        param_name: cl_device_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    /* Context APIs */
    pub fn clCreateContext(
        properties: *const cl_context_properties,
        num_devices: cl_uint,
        devices: *const cl_device_id,
        pfn_notify: extern fn (*const libc::c_char, *const libc::c_void, libc::size_t, *mut libc::c_void),
        user_data: *mut libc::c_void,
        errcode_ret: *mut cl_int) -> cl_context;

    pub fn clCreateContextFromType(
        properties: *mut cl_context_properties,
        device_type: cl_device_type,
        pfn_notify: extern fn (*mut libc::c_char, *mut libc::c_void, libc::size_t, *mut libc::c_void),
        user_data: *mut libc::c_void,
        errcode_ret: *mut cl_int) -> cl_context;

    pub fn clRetainContext(context: cl_context) -> cl_int;

    pub fn clReleaseContext(context: cl_context) -> cl_int;

    pub fn clGetContextInfo(
        context: cl_context,
        param_name: cl_context_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    /* Command Queue APIs */
    pub fn clCreateCommandQueue(
        context: cl_context,
        device: cl_device_id,
        properties: cl_command_queue_properties,
        errcode_ret: *mut cl_int) -> cl_command_queue;

    pub fn clRetainCommandQueue(command_queue: cl_command_queue) -> cl_int;

    pub fn clReleaseCommandQueue(command_queue: cl_command_queue) -> cl_int;

    pub fn clGetCommandQueueInfo(
        command_queue: cl_command_queue,
        param_name: cl_command_queue_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    /* Memory Object APIs */
    pub fn clCreateBuffer(
        context: cl_context,
        flags: cl_mem_flags,
        size: libc::size_t,
        host_ptr: *mut libc::c_void,
        errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clCreateSubBuffer(
        buffer: cl_mem,
        flags: cl_mem_flags,
        buffer_create_type: cl_buffer_create_type,
        buffer_create_info: *mut libc::c_void,
        errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clCreateImage2D(
        context: cl_context,
        flags: cl_mem_flags,
        image_format: *mut cl_image_format,
        image_width: libc::size_t,
        image_height: libc::size_t,
        image_row_pitch: libc::size_t,
        host_ptr: *mut libc::c_void,
        errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clCreateImage3D(
        context: cl_context,
        flags: cl_mem_flags,
        image_format: *mut cl_image_format,
        image_width: libc::size_t,
        image_height: libc::size_t,
        image_depth: libc::size_t,
        image_row_pitch: libc::size_t,
        image_depth: libc::size_t,
        image_row_pitch: libc::size_t,
        image_slice_pitch: libc::size_t,
        host_ptr: *mut libc::c_void,
        errcode_ret: *mut cl_int) -> cl_mem;

    pub fn clRetainMemObject(memobj: cl_mem) -> cl_int;

    pub fn clReleaseMemObject(memobj: cl_mem) -> cl_int;

    pub fn clGetSupportedImageFormats(
        context: cl_context,
        flags: cl_mem_flags,
        image_type: cl_mem_object_type,
        num_entries: cl_uint,
        image_formats: *mut cl_image_format,
        num_image_formats: *mut cl_uint) -> cl_int;

    pub fn clGetMemObjectInfo(
        memobj: cl_mem,
        param_name: cl_mem_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    pub fn clGetImageInfo(
        image: cl_mem,
        param_name: cl_image_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    pub fn clSetMemObjectDestructorCallback(
        memobj: cl_mem,
        pfn_notify: extern fn (cl_mem, *mut libc::c_void),
        user_data: *mut libc::c_void) -> cl_int;


    /*mut * Sampler APIs */
    pub fn clCreateSampler(
        context: cl_context,
        normalize_coords: cl_bool,
        addressing_mode: cl_addressing_mode,
        filter_mode: cl_filter_mode,
        errcode_ret: *mut cl_int) -> cl_sampler;

    pub fn clRetainSampler(sampler: cl_sampler) -> cl_int;

    pub fn clReleaseSampler(sampler: cl_sampler) ->cl_int;

    pub fn clGetSamplerInfo(
        sampler: cl_sampler,
        param_name: cl_sampler_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;


    /* Program Object APIs */
    pub fn clCreateProgramWithSource(
        context: cl_context,
        count: cl_uint,
        strings: *const *const libc::c_char,
        lengths: *const libc::size_t,
        errcode_ret: *mut cl_int) -> cl_program;

    pub fn clCreateProgramWithBinary(
        context: cl_context,
        num_devices: cl_uint,
        device_list: *const cl_device_id,
        lengths: *const libc::size_t,
        binaries: *const *const libc::c_uchar,
        binary_status: *mut cl_int,
        errcode_ret: *mut cl_int) -> cl_program;

    pub fn clRetainProgram(program: cl_program) -> cl_int;

    pub fn clReleaseProgram(program: cl_program) -> cl_int;

    pub fn clBuildProgram(
        program: cl_program,
        num_devices: cl_uint,
        device_list: *const cl_device_id,
        options: *const libc::c_char,
        pfn_notify: extern fn (cl_program, *mut libc::c_void),
        user_data: *mut libc::c_void) -> cl_int;

    pub fn clUnloadCompiler() -> cl_int;

    pub fn clGetProgramInfo(
        program: cl_program,
        param_name: cl_program_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    pub fn clGetProgramBuildInfo(
        program: cl_program,
        device: cl_device_id,
        param_name: cl_program_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    /* Kernel Object APIs */
    pub fn clCreateKernel(
        program: cl_program,
        kernel_name: *const libc::c_char,
        errcode_ret: *mut cl_int) -> cl_kernel;

    pub fn clCreateKernelsInProgram(
        program: cl_program,
        num_kernels: cl_uint,
        kernels: *mut cl_kernel,
        num_kernels_ret: *mut cl_uint) -> cl_int;

    pub fn clRetainKernel(kernel: cl_kernel) -> cl_int;

    pub fn clReleaseKernel(kernel: cl_kernel) -> cl_int;

    pub fn clSetKernelArg(
        kernel: cl_kernel,
        arg_index: cl_uint,
        arg_size: libc::size_t,
        arg_value: *const libc::c_void) -> cl_int;

    pub fn clGetKernelInfo(
        kernel: cl_kernel,
        param_name: cl_kernel_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    pub fn clGetKernelWorkGroupInfo(
        kernel: cl_kernel,
        device: cl_device_id,
        param_name: cl_kernel_work_group_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;


    /* Event Object APIs */
    pub fn clWaitForEvents(
        num_events: cl_uint,
        event_list: *const cl_event) -> cl_int;

    pub fn clGetEventInfo(
        event: cl_event,
        param_name: cl_event_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;

    pub fn clCreateUserEvent(
        context: cl_context,
        errcode_ret: *mut cl_int) -> cl_event;

    pub fn clRetainEvent(event: cl_event) -> cl_int;

    pub fn clReleaseEvent(event: cl_event) -> cl_int;

    pub fn clSetUserEventStatus(
        event: cl_event,
        execution_status: cl_int) -> cl_int;

    pub fn clSetEventCallback(
        event: cl_event,
        command_exec_callback_type: cl_int,
        pfn_notify: extern fn (cl_event, cl_int, *mut libc::c_void),
        user_data: *mut libc::c_void) -> cl_int;


    /* Profiling APIs */
    pub fn clGetEventProfilingInfo(
        event: cl_event,
        param_name: cl_profiling_info,
        param_value_size: libc::size_t,
        param_value: *mut libc::c_void,
        param_value_size_ret: *mut libc::size_t) -> cl_int;


    /* Flush and Finish APIs */
    pub fn clFlush(command_queue: cl_command_queue) -> cl_int;

    pub fn clFinish(command_queue: cl_command_queue) -> cl_int;

    /* Enqueued Commands APIs */
    pub fn clEnqueueReadBuffer(
        command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_read: cl_bool,
        offset: libc::size_t,
        cb: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueReadBufferRect(
        command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_read: cl_bool,
        buffer_origin: *mut libc::size_t,
        host_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        buffer_row_pitch: libc::size_t,
        buffer_slice_pitch: libc::size_t,
        host_row_pitch: libc::size_t,
        host_slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueWriteBuffer(
        command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_write: cl_bool,
        offset: libc::size_t,
        cb: libc::size_t,
        ptr: *const libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueWriteBufferRect(
        command_queue: cl_command_queue,
        blocking_write: cl_bool,
        buffer_origin: *mut libc::size_t,
        host_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        buffer_row_pitch: libc::size_t,
        buffer_slice_pitch: libc::size_t,
        host_row_pitch: libc::size_t,
        host_slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyBuffer(
        command_queue: cl_command_queue,
        src_buffer: cl_mem,
        dst_buffer: cl_mem,
        src_offset: libc::size_t,
        dst_offset: libc::size_t,
        cb: libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyBufferRect(
        command_queue: cl_command_queue,
        src_buffer: cl_mem,
        dst_buffer: cl_mem,
        src_origin: *mut libc::size_t,
        dst_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        src_row_pitch: libc::size_t,
        src_slice_pitch: libc::size_t,
        dst_row_pitch: libc::size_t,
        dst_slice_pitch: libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueReadImage(
        command_queue: cl_command_queue,
        image: cl_mem,
        blocking_read: cl_bool,
        origin: *mut libc::size_t,
        region: *mut libc::size_t,
        row_pitch: libc::size_t,
        slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueWriteImage(
        command_queue: cl_command_queue,
        image: cl_mem,
        blocking_write: cl_bool,
        origin: *mut libc::size_t,
        region: *mut libc::size_t,
        input_row_pitch: libc::size_t,
        input_slice_pitch: libc::size_t,
        ptr: *mut libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyImage(
        command_queue: cl_command_queue,
        src_image: cl_mem,
        dst_image: cl_mem,
        src_origin: *mut libc::size_t,
        dst_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyImageToBuffer(
        command_queue: cl_command_queue,
        src_image: cl_mem,
        dst_buffer: cl_mem,
        src_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        dst_offset: libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueCopyBufferToImage(
        command_queue: cl_command_queue,
        src_buffer: cl_mem,
        dst_image: cl_mem,
        src_offset: libc::size_t,
        dst_origin: *mut libc::size_t,
        region: *mut libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueMapBuffer(
        command_queue: cl_command_queue,
        buffer: cl_mem,
        blocking_map: cl_bool,
        map_flags: cl_map_flags,
        offset: libc::size_t,
        cb: libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event,
        errorcode_ret: *mut cl_int);

    pub fn clEnqueueMapImage(
        command_queue: cl_command_queue,
        image: cl_mem,
        blocking_map: cl_bool,
        map_flags: cl_map_flags,
        origin: *mut libc::size_t,
        region: *mut libc::size_t,
        image_row_pitch: libc::size_t,
        image_slice_pitch: libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event,
        errorcode_ret: *mut cl_int);

    pub fn clEnqueueUnmapMemObject(
        command_queue: cl_command_queue,
        memobj: cl_mem,
        mapped_ptr: *mut libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueNDRangeKernel(
        command_queue: cl_command_queue,
        kernel: cl_kernel,
        work_dim: cl_uint,
        global_work_offset: *const libc::size_t,
        global_work_size: *const libc::size_t,
        local_work_size: *const libc::size_t,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueTask(
        command_queue: cl_command_queue,
        kernel: cl_kernel,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueNativeKernel(
        command_queue: cl_command_queue,
        user_func: extern fn (*mut libc::c_void),
        args: *mut libc::c_void,
        cb_args: libc::size_t,
        num_mem_objects: cl_uint,
        mem_list: *const cl_mem,
        args_mem_loc: *const *const libc::c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const cl_event,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueMarker(
        command_queue: cl_command_queue,
        event: *mut cl_event) -> cl_int;

    pub fn clEnqueueWaitForEvents(
        command_queue: cl_command_queue,
        num_events: cl_uint,
        event_list: *mut cl_event) -> cl_int;

    pub fn clEnqueueBarrier(command_queue: cl_command_queue) -> cl_int;

    /* Extension function access
     *
     * Returns the extension function address for the given function name,
     * or NULL if a valid function can not be found. The client must
     * check to make sure the address is not NULL, before using or
     * or calling the returned function address.
     */
    pub fn clGetExtensionFunctionAddress(func_name: *const libc::c_char) -> *mut libc::c_void;
}

pub fn check(status: cl_int, message: String) -> Option<FrameworkError> {
    if status != CLStatus::CL_SUCCESS as cl_int {
        return Some(FrameworkError {
            code: status,
            id: CLStatus::from_i32(status).unwrap().to_string(),
            message: message,
        });
    }
    None
}
