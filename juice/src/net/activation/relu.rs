use crate::co::IBackend;
use crate::conn;
use crate::net::{Context, Descriptor, Layer, LayerBackend};

#[derive(Debug, Clone)]
pub struct Relu {
    descriptor: Descriptor,
}

impl Relu {
    pub fn new(mut descriptor: Descriptor) -> Self {
        assert_eq!(descriptor.inputs().len(), 1); // Should only be one input.

        descriptor.add_output(descriptor.input(0).unit_shape().clone());

        Relu {
            descriptor: descriptor,
        }
    }
}

impl Layer for Relu {
    fn compute_output(&self, backend: &dyn LayerBackend, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.acquire_data(self.descriptor.output(0));
        backend
            .relu(&input.borrow(), &mut output.borrow_mut())
            .unwrap();
    }

    fn compute_gradients(&self, backend: &dyn LayerBackend, context: &mut Context) {
        let input = context.get_data(self.descriptor.input(0));
        let output = context.get_data(self.descriptor.output(0));
        let output_gradient = context.get_data_gradient(self.descriptor.output(0));
        let input_gradient = context.acquire_data_gradient(self.descriptor.input(0));
        backend
            .relu_grad(
                &output.borrow(),
                &output_gradient.borrow(),
                &input.borrow(),
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