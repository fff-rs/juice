# pointer-tracker

A little helper crate for those suffering the chores of ptr and ptr to ptr
and sometimes even ptr to ptr to ptr handling in FFI bindings.

## Concept

Should be used inside the lower level FFI bindings which merely conver result types
to validate pointers being valid which in most cases implies allocated by a certain
allocator.

Internally this is nothing more than a `HashSet` and verification with some helpers
and `Sync` access.