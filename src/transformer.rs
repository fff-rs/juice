use blob::Blob;

/// The Transformer Trait
///
/// Gets implemented for all Transformable Data Types.
/// Allows all Transformable Data Types to get transformed into a `Blob`.
pub trait Transformer {

    /// Transforms non-numeric data into a numeric `Blob`
    ///
    /// The shape attribute is used to controll the dimensions/shape of the Blob.
    /// It returns an Error, when the expected capacity (defined by the shape) differs, from the
    /// observed one.
    fn transform(&self, shape: Vec<isize>) -> Result<Box<Blob<f32>>, TransformerError> {
        let mut blob = Box::new(Blob::of_shape(shape));
        match self.write_into_blob_data(blob.mutable_cpu_data()) {
            Ok(_) => Ok(blob),
            Err(e) => Err(e)
        }
    }

    /// Transforms the non-numeric data into a numeric `Vec`
    fn transform_to_vec(&self) -> Vec<f32>;

    /// Writes into `Blob`s' data
    fn write_into_blob_data(&self, blob_data: &mut Vec<f32>) -> Result<(), TransformerError> {
        let data = Box::new(self.transform_to_vec());
        if blob_data.capacity() == data.capacity() {
            for v in data.iter() {
                blob_data.push(*v);
            }
            Ok(())
        } else {
            Err(TransformerError::InvalidShape)
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// The Transformer Errors
pub enum TransformerError {
    /// When the speficied shape capacitiy differs from the actual capacity of the numeric Vec
    InvalidShape
}
