//! Provides BLAS for a Native backend.

use frameworks::native::{Function, Binary};
use memory::MemoryType;
use libraries::blas::*;

impl IBlasBinary for Binary {
    type Dot = Function;

    fn dot(&self) -> Self::Dot {
        self.blas_dot
    }
}

impl IOperationDot for Function {
    fn compute<T>(&self, x: &MemoryType, y: &MemoryType, result: &mut MemoryType) -> Result<(), Error> {
        let x_slice = try!(x.as_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `x`.")))).as_slice::<f32>();
        let y_slice = try!(y.as_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `y`.")))).as_slice::<f32>();
        let mut r_slice = try!(result.as_mut_native().ok_or(Error::InvalidArgument(format!("Unable to receive native memory for `result`.")))).as_mut_slice::<f32>();
        let mut tmp = vec![0f32];
        for (i, val) in x_slice.iter().enumerate() {
            tmp.insert(i, y_slice[i] * val);
        }
        r_slice[0] = tmp.iter().fold(0f32, |mut sum, x| {sum += *x; sum});
        Ok(())
    }
}
