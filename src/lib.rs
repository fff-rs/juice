//! Cuticula provides convenient and universal Machine Learning Encoding
//! funcionality for non-numeric data types such as: `Strings`, `Images` and
//! `Audio`.
extern crate image;
extern crate murmurhash3;

pub use transformer::ImageTransformer;
pub use transformer::StringTransformer;
pub use transformer::Transformer;

mod transformer;
