use std::env::var;
use std::ascii::AsciiExt;

fn main() {
    // Format: CARGO_BLAS_IMPL = (STATIC|DYLIB)_(BLAS_NAME)
    let lib = match var("CARGO_BLAS_IMPL") {
        Ok(s) => s,
        Err(_) => "DYLIB_BLAS".into(),
    };

    let mut pieces = lib.split("_");
    let link_type = match pieces.next() {
        Some(t) => {
            if t == "STATIC" {
                "static"
            } else if t == "DYLIB" {
                "dylib"
            } else {
                panic!("Link type must be either STATIC or DYLIB")
            }
        },
        None => panic!("Format error"),
    };
    let link_name = match pieces.next() {
        Some(t) => t.to_ascii_lowercase(),
        None => panic!("Format error"),
    };

    println!("cargo:rustc-link-lib={}={}", link_type, link_name);
}
