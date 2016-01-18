The file `src/generated.rs` was created with [bindgen](https://github.com/crabtw/rust-bindgen) using the command:

```sh
# /opt/cuda/include/cblas_v2.h    - the path of the cublas header
# -I /usr/lib/clang/3.7.0/include - makes sure clang headers are found; changes with clang version
### match statements are used so that only the neccessary parts are generated
# -match cublas                   - generates the bindings for cublas
# -l cublas                       - generates the link statement in the output file
# -o src/generated.rs             - output to src/generated.rs
bindgen /opt/cuda/include/cublas_v2.h -I /usr/lib/clang/3.7.0/include -match cublas -l cublas -o template/generated.gen.rs
# this helps us rexport the enum instead of aliasing them
sed -i '/pub type/d' template/generated.gen.rs
sed -i 's/Clone/PartialEq, Debug, Clone/g' template/generated.gen.rs
cat template/generated.header.rs template/generated.gen.rs > src/generated.rs
```
