# pointer-tracker

A little helper crate for those suffering the chores of ptr and ptr to ptr
and sometimes even ptr to ptr to ptr handling in FFI bindings.

## Concept

Should be used inside the lower level FFI bindings which merely conver result types
to validate pointers being valid which in most cases implies allocated by a certain
allocator.

Internally this is nothing more than a `HashSet` and verification with some helpers
and `Sync` access.

## Usage

Since the macro `ptr_origin_tracker::setup!(X)` does impl a trait for the pointer `*mut X`
it must reside in the same crate as it is generated, which is commonly `-sys`.
Exposing the whole tracker module via `pub use ptr_origin_tracker as tracker;` is the most ergonomic way,
where actually tracking can then be achieved as needed in the wrapping `ffi`/rustic-`API` layer.