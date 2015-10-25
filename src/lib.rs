//! Cuticula provides convenient and universal Machine Learning Transformer
//! for non-numeric data types such as: `Strings`, `Images` and `Audio`.
#![feature(plugin)]
#![plugin(clippy)]
extern crate image as image_lib;
extern crate murmurhash3 as murmur3;

pub use image::Image;
pub use word::Word;
pub use transformer::Transformer;

pub use modifier::Set;

pub mod transformer;
pub mod image;
pub mod word;

/// Re-exports from the Modifier crate.
pub mod modifier {
    extern crate modifier as modifier_lib;
    pub use self::modifier_lib::*;
}
