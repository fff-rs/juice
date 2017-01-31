use std::env::var;

fn main() {
    let link_type = env_or_default("CARGO_BLAS_TYPE", "dylib");
    let link_name = env_or_default("CARGO_BLAS", "openblas");

    println!("cargo:rustc-link-lib={}={}", link_type, link_name);
}

fn env_or_default(var_name: &str, default: &str) -> String {
    match var(var_name) {
        Ok(s) => s,
        Err(_) => default.into(),
    }
}
