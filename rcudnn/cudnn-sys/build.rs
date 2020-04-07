use pkg_config;
use std::env;

fn main() {
    let lib_dir = env::var("CUDNN_LIB_DIR").ok();
    let include_dir = env::var("CUDNN_INCLUDE_DIR").ok();

    if lib_dir.is_none() && include_dir.is_none() {
        if let Ok(info) = pkg_config::find_library("cudart") {
            // avoid empty include paths as they are not supported by GCC
            if !info.include_paths.is_empty() {
                let paths = env::join_paths(info.include_paths).unwrap();
                println!("cargo:include={}", paths.to_str().unwrap());
            }
        }
        if let Ok(info) = pkg_config::find_library("cuda") {
            // avoid empty include paths as they are not supported by GCC
            if !info.include_paths.is_empty() {
                let paths = env::join_paths(info.include_paths).unwrap();
                println!("cargo:include={}", paths.to_str().unwrap());
            }
        }
        if let Ok(info) = pkg_config::find_library("cudnn") {
            // avoid empty include paths as they are not supported by GCC
            if !info.include_paths.is_empty() {
                let paths = env::join_paths(info.include_paths).unwrap();
                println!("cargo:include={}", paths.to_str().unwrap());
            }
            return;
        }
    }

    let libs_env = env::var("CUDNN_LIBS").ok();
    let libs = match libs_env {
        Some(ref v) => v.split(':').collect(),
        None => vec!["cudnn", "cudart", "cuda"],
    };

    let mode = if env::var_os("CUDNN_STATIC").is_some() {
        "static"
    } else {
        "dylib"
    };

    if let Some(lib_dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }

    for lib in libs {
        println!("cargo:rustc-link-lib={}={}", mode, lib);
    }

    if let Some(include_dir) = include_dir.clone() {
        println!("cargo:include={}", include_dir);
    }

    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature="generate")]
    {
        println!("cargo:warning=Running bindgen(cudnn-sys), make sure to have all required host libs installed!");

        use std::path::PathBuf;

        let include_dir = include_dir
            .unwrap_or_else(|| String::from("/usr/include/cuda"));

        let bindings = bindgen::Builder::default()
            .rust_target(bindgen::RustTarget::Stable_1_40)
            .raw_line(
                r"
//! Defines the FFI for CUDA cuDNN.
//!
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
            ",
            )
            .ctypes_prefix("::libc")
            .size_t_is_usize(true)
            .clang_arg("-I")
            .clang_arg(include_dir)
            .header("wrapper.h")
            .rustified_enum("cudnn[A-Za-z]+_t")
            .rustified_enum("cudaError")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks))
            .generate()
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from("src").join("generated.rs");
        bindings
            .write_to_file(out_path)
            .expect("Couldn't write bindings!");
    }
}
