use crate::co::{IBackend};
use crate::conn;
use crate::net::{Context, Descriptor, Layer};

#[derive(Debug)]
pub struct LogSoftmax {
    descriptor: Descriptor,
}

impl LogSoftmax {
    pub fn new(mut descriptor: Descriptor) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should only be one input.

        descriptor.add_output(descriptor.input(0).unit_shape().clone());

        LogSoftmax {
            descriptor: descriptor,
        }
    }
}

impl<B: IBackend + conn::LogSoftmax<f32>> Layer<B> for LogSoftmax {
    fn compute_output(&self, context: &mut Context<B>) {
        let input = context.get_data(self.descriptor.input(0));
        let mut output = context.acquire_data(self.descriptor.output(0));
        context
            .backend()
            .log_softmax(&input.borrow(), &mut output.borrow_mut())
            .unwrap();
    }

    fn compute_gradients(&self, context: &mut Context<B>) {
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));
        let mut input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        context
            .backend()
            .log_softmax_grad(
                &output.borrow(),
                &output_gradient.borrow(),
                &mut input_gradient.borrow_mut(),
            )
            .unwrap();
    }

    fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    fn descriptor_mut(&mut self) -> &mut Descriptor {
        &mut self.descriptor
    }
}
