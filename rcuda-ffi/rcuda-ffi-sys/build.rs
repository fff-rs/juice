use pkg_config;
use std::env;

fn main() {
    let lib_dir = env::var("CUDA_HOME").ok();
    let include_dir = env::var("CUBLAS_INCLUDE_DIR").ok();

    cc::Build::new()
        .cuda(true)
        .include(lib_dir.unwrap())
        .include(include_dir.unwrap())
        .object("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\lib\\x64\\cudart.lib")
        .flag("-gencode").flag("arch=compute_52,code=sm_52")
        // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode").flag("arch=compute_53,code=sm_53")
        // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_61,code=sm_61")
        // Generate code for Pascal (Tesla P100).
        .flag("-gencode").flag("arch=compute_60,code=sm_60")
        // Generate code for Pascal (Jetson TX2).
        .flag("-gencode").flag("arch=compute_62,code=sm_62")
        .files(&["src/embedding/gather.cu", "src/utils/batchStridedSum.cu"])
        .compile("libcu_gather.a");
}
