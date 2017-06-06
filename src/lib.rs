//! Cuticula provides convenient and universal Machine Learning Transformer
//! for non-numeric data types such as: `Strings`, `Images` and `Audio`.
#![cfg_attr(lint, feature(plugin))]
#![cfg_attr(lint, plugin(clippy))]
#![allow(dead_code)]
#![deny(missing_docs,
        missing_debug_implementations, missing_copy_implementations,
        trivial_casts, trivial_numeric_casts,
        unsafe_code, unused_import_braces, unused_qualifications)]

extern crate collenchyma as co;
extern crate murmurhash3 as murmur3;
extern crate image as image_crate;

pub use image::Image;
pub use word::Word;
pub use transformer::Transformer;

pub use modifier::Set;

/// Transformer
pub mod transformer;
/// The Image Struct and its Modifiers
pub mod image;
/// The Word Struct and its Modifiers
pub mod word;


/// Re-exports from the modifier crate.
pub mod modifier {
    extern crate modifier as modifier_crate;
    pub use self::modifier_crate::*;
}
