//! Provides a operation on native CPU.

use operation::IOperation;

#[derive(Debug, Copy, Clone)]
/// Defines a host CPU operation.
pub struct Function {
    id: isize,
}

impl Function {
    /// Initializes a native CPU hardware.
    pub fn new(id: isize) -> Function {
        Function { id: id }
    }
}

impl IOperation for Function {}
