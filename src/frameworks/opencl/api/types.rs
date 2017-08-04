//! Provides the Foreign Type Interface for OpenCL.
#![allow(non_camel_case_types, dead_code)]
#![allow(missing_docs, missing_debug_implementations, missing_copy_implementations)]

use libc;
use std::fmt;

/* Opaque types */
pub type platform_id                 = *mut libc::c_void;
pub type device_id                   = *mut libc::c_void;
pub type context_id                  = *mut libc::c_void;
pub type queue_id                    = *mut libc::c_void;
pub type memory_id                   = *mut libc::c_void;
pub type program                     = *mut libc::c_void;
pub type kernel_id                   = *mut libc::c_void;
pub type event                       = *mut libc::c_void;
pub type sampler                     = *mut libc::c_void;

/* Scalar types */
pub type short                       = i16;
pub type ushort                      = u16;
pub type int                         = i32;
pub type uint                        = u32;
pub type long                        = i64;
pub type ulong                       = u64;
pub type half                        = u16;
pub type float                       = f32;
pub type double                      = f64;

pub type boolean                     = uint;
pub type bitfield                    = ulong;
pub type device_type                 = bitfield;
pub type platform_info               = uint;
pub type device_info                 = uint;
pub type device_fp_config            = bitfield;
pub type device_mem_cache_type       = uint;
pub type device_local_mem_type       = uint;
pub type device_exec_capabilities    = bitfield;
pub type command_queue_properties    = bitfield;

pub type context_properties          = libc::intptr_t;
pub type context_info                = uint;
pub type command_queue_info          = uint;
pub type channel_order               = uint;
pub type channel_type                = uint;
pub type mem_flags                   = bitfield;
pub type mem_object_type             = uint;
pub type mem_info                    = uint;
pub type image_info                  = uint;
pub type buffer_create_type          = uint;
pub type addressing_mode             = uint;
pub type filter_mode                 = uint;
pub type sampler_info                = uint;
pub type map_flags                   = bitfield;
pub type program_info                = uint;
pub type program_build_info          = uint;
pub type build_status                = int;
pub type kernel_info                 = uint;
pub type kernel_work_group_info      = uint;
pub type event_info                  = uint;
pub type command_type                = uint;
pub type profiling_info              = uint;

#[repr(C)]
pub struct image_format {
    image_channel_order:        channel_order,
    image_channel_data_type:    channel_type
}

pub struct buffer_region {
    origin:     libc::size_t,
    size:       libc::size_t
}


enum_from_primitive! {
/// OpenCL error codes.
#[derive(PartialEq, Debug)]
#[repr(C)]
pub enum Status {
    SUCCESS = 0,
    DEVICE_NOT_FOUND = -1,
    DEVICE_NOT_AVAILABLE = -2,
    COMPILER_NOT_AVAILABLE = -3,
    MEM_OBJECT_ALLOCATION_FAILURE = -4,
    OUT_OF_RESOURCES = -5,
    OUT_OF_HOST_MEMORY = -6,
    PROFILING_INFO_NOT_AVAILABLE = -7,
    MEM_COPY_OVERLAP = -8,
    IMAGE_FORMAT_MISMATCH = -9,
    IMAGE_FORMAT_NOT_SUPPORTED = -10,
    BUILD_PROGRAM_FAILURE = -11,
    MAP_FAILURE = -12,
    MISALIGNED_SUB_BUFFER_OFFSET = -13,
    EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14,
    INVALID_VALUE = -30,
    INVALID_DEVICE_TYPE = -31,
    INVALID_PLATFORM = -32,
    INVALID_DEVICE = -33,
    INVALID_CONTEXT = -34,
    INVALID_QUEUE_PROPERTIES = -35,
    INVALID_COMMAND_QUEUE = -36,
    INVALID_HOST_PTR = -37,
    INVALID_MEM_OBJECT = -38,
    INVALID_IMAGE_FORMAT_DESCRIPTOR = -39,
    INVALID_IMAGE_SIZE = -40,
    INVALID_SAMPLER = -41,
    INVALID_BINARY = -42,
    INVALID_BUILD_OPTIONS = -43,
    INVALID_PROGRAM = -44,
    INVALID_PROGRAM_EXECUTABLE = -45,
    INVALID_KERNEL_NAME = -46,
    INVALID_KERNEL_DEFINITION = -47,
    INVALID_KERNEL = -48,
    INVALID_ARG_INDEX = -49,
    INVALID_ARG_VALUE = -50,
    INVALID_ARG_SIZE = -51,
    INVALID_KERNEL_ARGS = -52,
    INVALID_WORK_DIMENSION = -53,
    INVALID_WORK_GROUP_SIZE = -54,
    INVALID_WORK_ITEM_SIZE = -55,
    INVALID_GLOBAL_OFFSET = -56,
    INVALID_EVENT_WAIT_LIST = -57,
    INVALID_EVENT = -58,
    INVALID_OPERATION = -59,
    INVALID_GL_OBJECT = -60,
    INVALID_BUFFER_SIZE = -61,
    INVALID_MIP_LEVEL = -62,
    INVALID_GLOBAL_WORK_SIZE = -63,
    INVALID_PROPERTY = -64,
    PLATFORM_NOT_FOUND_KHR = -1001,
}
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/* OpenCL Version */
pub static CL_VERSION_1_0:                               boolean = 1;
pub static CL_VERSION_1_1:                               boolean = 1;

/* cl_bool */
pub static CL_FALSE:                                     boolean = 0;
pub static CL_TRUE:                                      boolean = 1;

/* cl_platform_info */
pub static CL_PLATFORM_PROFILE:                          uint = 0x0900;
pub static CL_PLATFORM_VERSION:                          uint = 0x0901;
pub static CL_PLATFORM_NAME:                             uint = 0x0902;
pub static CL_PLATFORM_VENDOR:                           uint = 0x0903;
pub static CL_PLATFORM_EXTENSIONS:                       uint = 0x0904;

/* cl_device_type - bitfield */
pub enum DeviceType {
    DEFAULT(bitfield),
    CPU(bitfield),
    GPU(bitfield),
    ACCELERATOR(bitfield),
    ALL(bitfield),
}
pub static DEVICE_TYPE: [DeviceType; 5] = [
    DeviceType::DEFAULT(1),
    DeviceType::CPU(1<<1),
    DeviceType::GPU(1<<2),
    DeviceType::ACCELERATOR(1<<3),
    DeviceType::ALL(0xFFFFFFFF)
];
pub const CL_DEVICE_TYPE_DEFAULT:                       bitfield = 1;
pub const CL_DEVICE_TYPE_CPU:                           bitfield = 1 << 1;
pub const CL_DEVICE_TYPE_GPU:                           bitfield = 1 << 2;
pub const CL_DEVICE_TYPE_ACCELERATOR:                   bitfield = 1 << 3;
pub const CL_DEVICE_TYPE_CUSTOM:                        bitfield = 1 << 4;
pub static CL_DEVICE_TYPE_ALL:                          bitfield = 0xFFFFFFFF;

/* cl_device_info */
pub static CL_DEVICE_TYPE:                               uint = 0x1000;
pub static CL_DEVICE_VENDOR_ID:                          uint = 0x1001;
pub static CL_DEVICE_MAX_COMPUTE_UNITS:                  uint = 0x1002;
pub static CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:           uint = 0x1003;
pub static CL_DEVICE_MAX_WORK_GROUP_SIZE:                uint = 0x1004;
pub static CL_DEVICE_MAX_WORK_ITEM_SIZES:                uint = 0x1005;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:        uint = 0x1006;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:       uint = 0x1007;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:         uint = 0x1008;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:        uint = 0x1009;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:       uint = 0x100A;
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:      uint = 0x100B;
pub static CL_DEVICE_MAX_CLOCK_FREQUENCY:                uint = 0x100C;
pub static CL_DEVICE_ADDRESS_BITS:                       uint = 0x100D;
pub static CL_DEVICE_MAX_READ_IMAGE_ARGS:                uint = 0x100E;
pub static CL_DEVICE_MAX_WRITE_IMAGE_ARGS:               uint = 0x100F;
pub static CL_DEVICE_MAX_MEM_ALLOC_SIZE:                 uint = 0x1010;
pub static CL_DEVICE_IMAGE2D_MAX_WIDTH:                  uint = 0x1011;
pub static CL_DEVICE_IMAGE2D_MAX_HEIGHT:                 uint = 0x1012;
pub static CL_DEVICE_IMAGE3D_MAX_WIDTH:                  uint = 0x1013;
pub static CL_DEVICE_IMAGE3D_MAX_HEIGHT:                 uint = 0x1014;
pub static CL_DEVICE_IMAGE3D_MAX_DEPTH:                  uint = 0x1015;
pub static CL_DEVICE_IMAGE_SUPPORT:                      uint = 0x1016;
pub static CL_DEVICE_MAX_PARAMETER_SIZE:                 uint = 0x1017;
pub static CL_DEVICE_MAX_SAMPLERS:                       uint = 0x1018;
pub static CL_DEVICE_MEM_BASE_ADDR_ALIGN:                uint = 0x1019;
pub static CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:           uint = 0x101A;
pub static CL_DEVICE_SINGLE_FP_CONFIG:                   uint = 0x101B;
pub static CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:              uint = 0x101C;
pub static CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:          uint = 0x101D;
pub static CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:              uint = 0x101E;
pub static CL_DEVICE_GLOBAL_MEM_SIZE:                    uint = 0x101F;
pub static CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:           uint = 0x1020;
pub static CL_DEVICE_MAX_CONSTANT_ARGS:                  uint = 0x1021;
pub static CL_DEVICE_LOCAL_MEM_TYPE:                     uint = 0x1022;
pub static CL_DEVICE_LOCAL_MEM_SIZE:                     uint = 0x1023;
pub static CL_DEVICE_ERROR_CORRECTION_SUPPORT:           uint = 0x1024;
pub static CL_DEVICE_PROFILING_TIMER_RESOLUTION:         uint = 0x1025;
pub static CL_DEVICE_ENDIAN_LITTLE:                      uint = 0x1026;
pub static CL_DEVICE_AVAILABLE:                          uint = 0x1027;
pub static CL_DEVICE_COMPILER_AVAILABLE:                 uint = 0x1028;
pub static CL_DEVICE_EXECUTION_CAPABILITIES:             uint = 0x1029;
pub static CL_DEVICE_QUEUE_PROPERTIES:                   uint = 0x102A;
pub const CL_DEVICE_NAME:                                uint = 0x102B;
pub static CL_DEVICE_VENDOR:                             uint = 0x102C;
pub static CL_DRIVER_VERSION:                            uint = 0x102D;
pub static CL_DEVICE_PROFILE:                            uint = 0x102E;
pub static CL_DEVICE_VERSION:                            uint = 0x102F;
pub static CL_DEVICE_EXTENSIONS:                         uint = 0x1030;
pub static CL_DEVICE_PLATFORM:                           uint = 0x1031;
/* 0x1032 reserved for CL_DEVICE_DOUBLE_FP_CONFIG */
/* 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG */
pub static CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:        uint = 0x1034;
pub static CL_DEVICE_HOST_UNIFIED_MEMORY:                uint = 0x1035;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:           uint = 0x1036;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:          uint = 0x1037;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:            uint = 0x1038;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:           uint = 0x1039;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:          uint = 0x103A;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:         uint = 0x103B;
pub static CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:           uint = 0x103C;
pub static CL_DEVICE_OPENCL_C_VERSION:                   uint = 0x103D;

/* cl_device_fp_config - bitfield */
pub static CL_FP_DENORM:                                 bitfield = 1;
pub static CL_FP_INF_NAN:                                bitfield = 1 << 1;
pub static CL_FP_ROUND_TO_NEAREST:                       bitfield = 1 << 2;
pub static CL_FP_ROUND_TO_ZERO:                          bitfield = 1 << 3;
pub static CL_FP_ROUND_TO_INF:                           bitfield = 1 << 4;
pub static CL_FP_FMA:                                    bitfield = 1 << 5;
pub static CL_FP_SOFT_FLOAT:                             bitfield = 1 << 6;

/* cl_device_mem_cache_type */
pub static CL_NONE:                                      uint = 0x0;
pub static CL_READ_ONLY_CACHE:                           uint = 0x1;
pub static CL_READ_WRITE_CACHE:                          uint = 0x2;

/* cl_device_local_mem_type */
pub static CL_LOCAL:                                     uint = 0x1;
pub static CL_GLOBAL:                                    uint = 0x2;

/* cl_device_exec_capabilities - bitfield */
pub static CL_EXEC_KERNEL:                               bitfield = 1;
pub static CL_EXEC_NATIVE_KERNEL:                        bitfield = 1 << 1;

/* cl_command_queue_properties - bitfield */
pub static CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE:       bitfield = 1;
pub static CL_QUEUE_PROFILING_ENABLE:                    bitfield = 1 << 1;

/* cl_context_info  */
pub const CL_CONTEXT_REFERENCE_COUNT:                   uint = 0x1080;
pub const CL_CONTEXT_DEVICES:                           uint = 0x1081;
pub const CL_CONTEXT_PROPERTIES:                        uint = 0x1082;
pub const CL_CONTEXT_NUM_DEVICES:                       uint = 0x1083;

/* cl_context_info + cl_context_properties */
pub const CL_CONTEXT_PLATFORM:                          libc::intptr_t = 0x1084;
pub const CL_CONTEXT_INTEROP_USER_SYNC:                 libc::intptr_t = 0x1085;

/* cl_command_queue_info */
pub static CL_QUEUE_CONTEXT:                             uint = 0x1090;
pub static CL_QUEUE_DEVICE:                              uint = 0x1091;
pub static CL_QUEUE_REFERENCE_COUNT:                     uint = 0x1092;
pub static CL_QUEUE_PROPERTIES:                          uint = 0x1093;

/* cl_mem_flags - bitfield */
pub static CL_MEM_READ_WRITE:                            bitfield = 1;
pub static CL_MEM_WRITE_ONLY:                            bitfield = 1 << 1;
pub static CL_MEM_READ_ONLY:                             bitfield = 1 << 2;
pub static CL_MEM_USE_HOST_PTR:                          bitfield = 1 << 3;
pub static CL_MEM_ALLOC_HOST_PTR:                        bitfield = 1 << 4;
pub static CL_MEM_COPY_HOST_PTR:                         bitfield = 1 << 5;

/* cl_channel_order */
pub static CL_R:                                         uint = 0x10B0;
pub static CL_A:                                         uint = 0x10B1;
pub static CL_RG:                                        uint = 0x10B2;
pub static CL_RA:                                        uint = 0x10B3;
pub static CL_RGB:                                       uint = 0x10B4;
pub static CL_RGBA:                                      uint = 0x10B5;
pub static CL_BGRA:                                      uint = 0x10B6;
pub static CL_ARGB:                                      uint = 0x10B7;
pub static CL_INTENSITY:                                 uint = 0x10B8;
pub static CL_LUMINANCE:                                 uint = 0x10B9;
pub static CL_RX:                                        uint = 0x10BA;
pub static CL_RGX:                                       uint = 0x10BB;
pub static CL_RGBX:                                      uint = 0x10BC;

/* cl_channel_type */
pub static CL_SNORM_INT8:                                uint = 0x10D0;
pub static CL_SNORM_INT16:                               uint = 0x10D1;
pub static CL_UNORM_INT8:                                uint = 0x10D2;
pub static CL_UNORM_INT16:                               uint = 0x10D3;
pub static CL_UNORM_SHORT_565:                           uint = 0x10D4;
pub static CL_UNORM_SHORT_555:                           uint = 0x10D5;
pub static CL_UNORM_INT_101010:                          uint = 0x10D6;
pub static CL_SIGNED_INT8:                               uint = 0x10D7;
pub static CL_SIGNED_INT16:                              uint = 0x10D8;
pub static CL_SIGNED_INT32:                              uint = 0x10D9;
pub static CL_UNSIGNED_INT8:                             uint = 0x10DA;
pub static CL_UNSIGNED_INT16:                            uint = 0x10DB;
pub static CL_UNSIGNED_INT32:                            uint = 0x10DC;
pub static CL_HALF_FLOAT:                                uint = 0x10DD;
pub static CL_FLOAT:                                     uint = 0x10DE;

/* cl_mem_object_type */
pub static CL_MEM_OBJECT_BUFFER:                         uint = 0x10F0;
pub static CL_MEM_OBJECT_IMAGE2D:                        uint = 0x10F1;
pub static CL_MEM_OBJECT_IMAGE3D:                        uint = 0x10F2;

/* cl_mem_info */
pub static CL_MEM_TYPE:                                  uint = 0x1100;
pub static CL_MEM_FLAGS:                                 uint = 0x1101;
pub static CL_MEM_SIZE:                                  uint = 0x1102;
pub static CL_MEM_HOST_PTR:                              uint = 0x1103;
pub static CL_MEM_MAP_COUNT:                             uint = 0x1104;
pub static CL_MEM_REFERENCE_COUNT:                       uint = 0x1105;
pub static CL_MEM_CONTEXT:                               uint = 0x1106;
pub static CL_MEM_ASSOCIATED_MEMOBJECT:                  uint = 0x1107;
pub static CL_MEM_OFFSET:                                uint = 0x1108;

/* cl_image_info */
pub static CL_IMAGE_FORMAT:                              uint = 0x1110;
pub static CL_IMAGE_ELEMENT_SIZE:                        uint = 0x1111;
pub static CL_IMAGE_ROW_PITCH:                           uint = 0x1112;
pub static CL_IMAGE_SLICE_PITCH:                         uint = 0x1113;
pub static CL_IMAGE_WIDTH:                               uint = 0x1114;
pub static CL_IMAGE_HEIGHT:                              uint = 0x1115;
pub static CL_IMAGE_DEPTH:                               uint = 0x1116;

/* cl_addressing_mode */
pub static CL_ADDRESS_NONE:                              uint = 0x1130;
pub static CL_ADDRESS_CLAMP_TO_EDGE:                     uint = 0x1131;
pub static CL_ADDRESS_CLAMP:                             uint = 0x1132;
pub static CL_ADDRESS_REPEAT:                            uint = 0x1133;
pub static CL_ADDRESS_MIRRORED_REPEAT:                   uint = 0x1134;

/* cl_filter_mode */
pub static CL_FILTER_NEAREST:                            uint = 0x1140;
pub static CL_FILTER_LINEAR:                             uint = 0x1141;

/* cl_sampler_info */
pub static CL_SAMPLER_REFERENCE_COUNT:                   uint = 0x1150;
pub static CL_SAMPLER_CONTEXT:                           uint = 0x1151;
pub static CL_SAMPLER_NORMALIZED_COORDS:                 uint = 0x1152;
pub static CL_SAMPLER_ADDRESSING_MODE:                   uint = 0x1153;
pub static CL_SAMPLER_FILTER_MODE:                       uint = 0x1154;

/* cl_map_flags - bitfield */
pub static CL_MAP_READ:                                  bitfield = 1;
pub static CL_MAP_WRITE:                                 bitfield = 1 << 1;

/* cl_program_info */
pub static CL_PROGRAM_REFERENCE_COUNT:                   uint = 0x1160;
pub static CL_PROGRAM_CONTEXT:                           uint = 0x1161;
pub static CL_PROGRAM_NUM_DEVICES:                       uint = 0x1162;
pub static CL_PROGRAM_DEVICES:                           uint = 0x1163;
pub static CL_PROGRAM_SOURCE:                            uint = 0x1164;
pub static CL_PROGRAM_BINARY_SIZES:                      uint = 0x1165;
pub static CL_PROGRAM_BINARIES:                          uint = 0x1166;

/* cl_program_build_info */
pub static CL_PROGRAM_BUILD_STATUS:                      uint = 0x1181;
pub static CL_PROGRAM_BUILD_OPTIONS:                     uint = 0x1182;
pub static CL_PROGRAM_BUILD_LOG:                         uint = 0x1183;

/* cl_build_status */
pub static CL_BUILD_SUCCESS:                             uint = 0;
pub static CL_BUILD_NONE:                                uint = (-1isize) as uint;
pub static CL_BUILD_ERROR:                               uint = -2isize as uint;
pub static CL_BUILD_IN_PROGRESS:                         uint = -3isize as uint;

/* cl_kernel_info */
pub static CL_KERNEL_FUNCTION_NAME:                      uint = 0x1190;
pub static CL_KERNEL_NUM_ARGS:                           uint = 0x1191;
pub static CL_KERNEL_REFERENCE_COUNT:                    uint = 0x1192;
pub static CL_KERNEL_CONTEXT:                            uint = 0x1193;
pub static CL_KERNEL_PROGRAM:                            uint = 0x1194;

/* cl_kernel_work_group_info */
pub static CL_KERNEL_WORK_GROUP_SIZE:                    uint = 0x11B0;
pub static CL_KERNEL_COMPILE_WORK_GROUP_SIZE:            uint = 0x11B1;
pub static CL_KERNEL_LOCAL_MEM_SIZE:                     uint = 0x11B2;
pub static CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: uint = 0x11B3;
pub static CL_KERNEL_PRIVATE_MEM_SIZE:                   uint = 0x11B4;

/* cl_event_info  */
pub static CL_EVENT_COMMAND_QUEUE:                       uint = 0x11D0;
pub static CL_EVENT_COMMAND_TYPE:                        uint = 0x11D1;
pub static CL_EVENT_REFERENCE_COUNT:                     uint = 0x11D2;
pub static CL_EVENT_COMMAND_EXECUTION_STATUS:            uint = 0x11D3;
pub static CL_EVENT_CONTEXT:                             uint = 0x11D4;

/* cl_command_type */
pub static CL_COMMAND_NDRANGE_KERNEL:                    uint = 0x11F0;
pub static CL_COMMAND_TASK:                              uint = 0x11F1;
pub static CL_COMMAND_NATIVE_KERNEL:                     uint = 0x11F2;
pub static CL_COMMAND_READ_BUFFER:                       uint = 0x11F3;
pub static CL_COMMAND_WRITE_BUFFER:                      uint = 0x11F4;
pub static CL_COMMAND_COPY_BUFFER:                       uint = 0x11F5;
pub static CL_COMMAND_READ_IMAGE:                        uint = 0x11F6;
pub static CL_COMMAND_WRITE_IMAGE:                       uint = 0x11F7;
pub static CL_COMMAND_COPY_IMAGE:                        uint = 0x11F8;
pub static CL_COMMAND_COPY_IMAGE_TO_BUFFER:              uint = 0x11F9;
pub static CL_COMMAND_COPY_BUFFER_TO_IMAGE:              uint = 0x11FA;
pub static CL_COMMAND_MAP_BUFFER:                        uint = 0x11FB;
pub static CL_COMMAND_MAP_IMAGE:                         uint = 0x11FC;
pub static CL_COMMAND_UNMAP_MEM_OBJECT:                  uint = 0x11FD;
pub static CL_COMMAND_MARKER:                            uint = 0x11FE;
pub static CL_COMMAND_ACQUIRE_GL_OBJECTS:                uint = 0x11FF;
pub static CL_COMMAND_RELEASE_GL_OBJECTS:                uint = 0x1200;
pub static CL_COMMAND_READ_BUFFER_RECT:                  uint = 0x1201;
pub static CL_COMMAND_WRITE_BUFFER_RECT:                 uint = 0x1202;
pub static CL_COMMAND_COPY_BUFFER_RECT:                  uint = 0x1203;
pub static CL_COMMAND_USER:                              uint = 0x1204;

/* command execution status */
pub static CL_COMPLETE:                                  uint = 0x0;
pub static CL_RUNNING:                                   uint = 0x1;
pub static CL_SUBMITTED:                                 uint = 0x2;
pub static CL_QUEUED:                                    uint = 0x3;

/* cl_buffer_create_type  */
pub static CL_BUFFER_CREATE_TYPE_REGION:                 uint = 0x1220;

/* cl_profiling_info  */
pub static CL_PROFILING_COMMAND_QUEUED:                  uint = 0x1280;
pub static CL_PROFILING_COMMAND_SUBMIT:                  uint = 0x1281;
pub static CL_PROFILING_COMMAND_START:                   uint = 0x1282;
pub static CL_PROFILING_COMMAND_END:                     uint = 0x1283;
