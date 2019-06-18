use co::prelude::*;
use co::plugin::numeric_helpers::*;

/// The Transformer Trait
///
/// Gets implemented for all Transformable Data Types.
/// Allows all Transformable Data Types to get transformed into a `Blob`.
pub trait Transformer {

    /// Transforms non-numeric data into a numeric `SharedTensor`
    ///
    /// The shape attribute is used to control the dimensions/shape of the Blob.
    /// It returns an Error, when the expected capacity (defined by the shape) differs from the
    /// observed one.
    fn transform(&self, shape: &[usize]) -> Result<SharedTensor<f32>, TransformerError> {
        let native_backend = Backend::<Native>::default().unwrap();
        let mut tensor = SharedTensor::<f32>::new(&shape);

        {
            let mut native_tensor = tensor.write_only(native_backend.device()).unwrap();
            Self::write_to_memory(&mut native_tensor, &self.transform_to_vec())?;
        }
        Ok(tensor)
    }

    /// Transforms the non-numeric data into a numeric `Vec`
    fn transform_to_vec(&self) -> Vec<f32>;

    /// Write into a native Coaster Memory.
    fn write_to_memory<T: NumCast + ::std::marker::Copy>(mem: &mut FlatBox, data: &[T]) -> Result<(), TransformerError> {
        Self::write_to_memory_offset(mem, data, 0)
    }

    /// Write into a native Coaster Memory with a offset.
    fn write_to_memory_offset<T: NumCast + ::std::marker::Copy>(mem: &mut FlatBox, data: &[T], offset: usize) -> Result<(), TransformerError> {
        let mut mem_buffer = mem.as_mut_slice::<f32>();
        if offset == 0 && mem_buffer.len() != data.len() {
            return Err(TransformerError::InvalidShape);
        }
        for (index, datum) in data.iter().enumerate() {
            let old_val = mem_buffer.get_mut(index + offset).ok_or(TransformerError::InvalidShape)?;
            *old_val = cast(*datum).unwrap();
        }
        Ok(())
    }
}

#[derive(Debug, Copy, Clone)]
/// The Transformer Errors
pub enum TransformerError {
    /// When the speficied shape capacitiy differs from the actual capacity of the numeric Vec
    InvalidShape,
    /// When The Image Pixel Buffer can't be converted to a RGB Image
    InvalidRgbPixels,
    /// When The Image Pixel Buffer can't be converted to a RGBA Image
    InvalidRgbaPixels,
    /// When The Image Pixel Buffer can't be converted to a greyscale Image
    InvalidLumaPixels,
    /// When The Image Pixel Buffer can't be converted to a greyscale Alpha Image
    InvalidLumaAlphaPixels,
}
