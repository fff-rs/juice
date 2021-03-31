//! Greenglas provides convenient and universal Machine Learning Transformer
//! for non-numeric data types such as: `Strings`, `Images` and `Audio`.
#![allow(dead_code)]
#![deny(
    clippy::missing_docs,
    clippy::missing_debug_implementations,
    clippy::missing_copy_implementations,
    clippy::trivial_casts,
    clippy::trivial_numeric_casts,
    clippy::unsafe_code,
    clippy::unused_import_braces,
    clippy::unused_qualifications,
    clippy::complexity
)]

extern crate coaster as co;
extern crate image as image_crate;
extern crate murmurhash3 as murmur3;

pub use crate::image::Image;
pub use crate::transformer::Transformer;
pub use crate::word::Word;

pub use crate::modifier::Set;

/// The Image Struct and its Modifiers
pub mod image;
/// Transformer
pub mod transformer;
/// The Word Struct and its Modifiers
pub mod word;

/// Re-exports from the modifier crate.
pub mod modifier {
    extern crate modifier as modifier_crate;
    pub use self::modifier_crate::*;
}
