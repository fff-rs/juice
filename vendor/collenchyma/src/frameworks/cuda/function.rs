//! Provides a Rust wrapper around Cuda's Function.

use operation::IOperation;

#[derive(Debug, Copy, Clone)]
/// Defines a Cuda Function.
///
/// A Function is Cuda's version of Collenchyma's [operation][operation].
/// [operation]: ../../operation/index.html
pub struct Function {
    id: isize,
}

impl Function {
    /// Initializes a new OpenCL device.
    pub fn from_isize(id: isize) -> Function {
        Function { id: id }
    }

    /*
    /// Initializes a new OpenCL device from its C type.
    //pub fn from_c(id: cl::kernel_id) -> Function {
        Function { id: id as isize }
    }

    /// Returns the id as its C type.
    pub fn id_c(&self) -> cl::kernel_id {
        self.id as cl::kernel_id
    }
    */
}

impl IOperation for Function {}
