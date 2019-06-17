extern crate bindgen;
extern crate pkg_config;
use std::env;
use std::path::PathBuf;

fn main() {
    let lib_dir = env::var("CUDNN_LIB_DIR").ok();
    let include_dir = env::var("CUDNN_INCLUDE_DIR").ok();

    if lib_dir.is_none() && include_dir.is_none() {
        if let Ok(info) = pkg_config::find_library("cudart") {
            // avoid empty include paths as they are not supported by GCC
            if info.include_paths.len() > 0 {
                let paths = env::join_paths(info.include_paths).unwrap();
                println!("cargo:include={}", paths.to_str().unwrap());
            }
        }
        if let Ok(info) = pkg_config::find_library("cuda") {
            // avoid empty include paths as they are not supported by GCC
            if info.include_paths.len() > 0 {
                let paths = env::join_paths(info.include_paths).unwrap();
                println!("cargo:include={}", paths.to_str().unwrap());
            }
        }
        if let Ok(info) = pkg_config::find_library("cudnn") {
            // avoid empty include paths as they are not supported by GCC
            if info.include_paths.len() > 0 {
                let paths = env::join_paths(info.include_paths).unwrap();
                println!("cargo:include={}", paths.to_str().unwrap());
            }
            return;
        }
    }

    let libs_env = env::var("CUDNN_LIBS").ok();
    let libs = match libs_env {
        Some(ref v) => v.split(":").collect(),
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

    if false {
        let bindings = bindgen::Builder::default()
            // Do not generate unstable Rust code that
            // requires a nightly rustc and enabling
            // unstable features.
            .rust_target(bindgen::RustTarget::Stable_1_19)
            .hide_type("max_align_t") // https://github.com/servo/rust-bindgen/issues/550
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
            .clang_arg("-I")
            .clang_arg(
                include_dir
                    .unwrap_or(String::from("/usr/include/cuda"))
                    .as_str(),
            )
            // The input header we would like to generate
            // bindings for.
            .header("wrapper.h")
            // Finish the builder and generate the bindings.
            .generate()
            // Unwrap the Result and panic on failure.
            .expect("Unable to generate bindings");

        let out_path = PathBuf::from("src");
        bindings
            .write_to_file(out_path.join("generated.rs"))
            .expect("Couldn't write bindings!");
    }
}
