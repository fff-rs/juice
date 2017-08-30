extern crate pkg_config;
use std::env;

fn main() {
    let variant = env::var("BLAS_VARIANT").unwrap_or("openblas".to_string());
    let lib_dir = env::var("BLAS_LIB_DIR").ok();
    let include_dir = env::var("BLAS_INCLUDE_DIR").ok();

    if lib_dir.is_none() && include_dir.is_none() {
        if let Ok(info) = pkg_config::find_library(variant.as_str()) {
            // avoid empty include paths as they are not supported by GCC
            if info.include_paths.len() > 0 {
                let paths = env::join_paths(info.include_paths).unwrap();
                println!("cargo:include={}", paths.to_str().unwrap());
            }
            return;
        }
    }

    let mode = if env::var_os("BLAS_STATIC").is_some() {
        "static"
    } else {
        "dylib"
    };

    if let Some(lib_dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    }

    println!("cargo:rustc-link-lib={}={}", mode, variant);

    if let Some(include_dir) = include_dir.clone() {
        println!("cargo:include={}", include_dir);
    }
}
