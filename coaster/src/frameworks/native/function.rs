//! Provides a operation on native CPU.

use operation::IOperation;

#[derive(Debug, Copy, Clone)]
/// Defines a host CPU operation.
pub struct Function;

impl Function {
    /// Initializes a native CPU hardware.
    pub fn new() -> Function {
        Function
    }
}

impl IOperation for Function {}
