use crate::co::{IBackend, ITensorDesc};
use crate::net::{Context, Descriptor, Layer, LayerError};
use crate::util::native_backend;

#[derive(Debug)]
pub struct NegativeLogLikelihood {
    descriptor: Descriptor,
}

impl NegativeLogLikelihood {
    pub fn new(descriptor: Descriptor) -> Self {
        assert_eq!(
            descriptor.inputs().len(),
            2,
            "NegativeLogLikelihood must take 2 inputs: probabilities and labels"
        );
        assert_eq!(
            descriptor.inputs()[1].unit_shape().size(),
            1,
            "Labels must be of [1] shape"
        );

        // Note that loss layers don't have outputs, since the result of loss computation is always
        // a single number which can't then be piped to other layers which expect data to have
        // shape [batch_size, ...]

        NegativeLogLikelihood { descriptor }
    }
}

impl<B: IBackend> Layer<B> for NegativeLogLikelihood {
    fn compute_output(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        // No output computation since loss layer doesn't have outputs.
        // It's main purpose is to start the backpropagation process by
        // computing the loss gradient with respect to net final output
        // in compute_gradients().
        Ok(())
    }

    fn compute_gradients(&self, backend: &B, context: &mut Context) -> Result<(), LayerError> {
        let probabilities_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        let labels = context.get_data(self.descriptor.input(1));

        let native = native_backend();
        let labels_data = labels.borrow();
        let native_labels = labels_data.read(native.device())?.as_slice::<f32>();
        let mut writable_gradient = vec![0f32; probabilities_gradient.borrow().desc().size()];

        let num_classes = self.descriptor.input(0).unit_shape().size();

        for (batch_n, &label_value) in native_labels.iter().enumerate() {
            let label_index = label_value as usize;
            assert!(
                label_index < num_classes,
                "Wrong label {} exceeding input size {}",
                label_index,
                num_classes
            );

            let index = (num_classes * batch_n) + label_index;
            writable_gradient[index] = -1f32;
        }
        crate::util::write_to_memory(
            probabilities_gradient.borrow_mut().write_only(native.device())?,
            &writable_gradient,
        );
        Ok(())
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}
