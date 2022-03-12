use std::cell::RefCell;
use std::collections::HashMap;
use std::iter;
use std::rc::Rc;

use crate::co::{IBackend, SharedTensor, TensorDesc};
use crate::net::{Inout, LearnableParamsLink};

/// Context stores data for a single invocation of a network (forward and/or backward),
/// which includes data passed between layers (at Junctions), loss function gradients with respect
/// to this data (again, at Junctions) and also loss function gradients with respect to the
/// learnable weights.
#[derive(Debug)]
pub struct Context<B: IBackend> {
    backend: Rc<B>,
    batch_size: usize,
    data: HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
    data_gradient: HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
    param_gradient: HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
}

impl<B: IBackend> Context<B> {
    pub fn new(backend: Rc<B>, batch_size: usize) -> Self {
        Context {
            backend: backend,
            batch_size: batch_size,
            data: HashMap::new(),
            data_gradient: HashMap::new(),
            param_gradient: HashMap::new(),
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    // Get data buffer for the given inout spec found inside the current scope.
    // Panics if this data buffer doesn't exist.
    pub fn get_data(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::get_inout_buffer(&self.data, inout, "data")
    }

    // Get data gradient buffer for the given inout spec found inside the current scope.
    // Panics if this data buffer doesn't exist.
    pub fn get_data_gradient(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::get_inout_buffer(&self.data_gradient, inout, "data gradient")
    }

    // Same as `get_data()` but will create the buffer on the fly if it doesn't exist.
    pub fn acquire_data(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::acquire_inout_buffer(&mut self.data, inout, self.batch_size, "data")
    }

    // Same as `get_data_gradient()` but will create the buffer on the fly if it doesn't exist.
    pub fn acquire_data_gradient(&mut self, inout: &Inout) -> Rc<RefCell<SharedTensor<f32>>> {
        Self::acquire_inout_buffer(&mut self.data_gradient, inout, self.batch_size, "data gradient")
    }

    // Get params gradient buffer for the given learnable params link.
    // Panics if this data buffer doesn't exist.
    pub fn get_params_gradient(
        &mut self,
        params: &LearnableParamsLink,
    ) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&params.params) as usize;
        match self.param_gradient.get(&key) {
            Some(data) => data.clone(),
            None => panic!("No params gradient for {:?}", params.params.borrow()),
        }
    }

    // Same as `get_params_gradient()` but will create the buffer on the fly if it doesn't exist.
    pub fn acquire_params_gradient(
        &mut self,
        params: &LearnableParamsLink,
    ) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&params.params) as usize;
        match self.param_gradient.get(&key) {
            Some(data) => return data.clone(),
            None => (),
        }

        println!("Creating params gradient for {:?}", params.params.borrow());

        let shape = params.params.borrow().data.desc().clone();
        let buffer = SharedTensor::<f32>::new(&shape);
        let buffer_rc =  Rc::new(RefCell::new(buffer));
        self.param_gradient.insert(key, buffer_rc.clone());
        buffer_rc
    }

    fn get_inout_buffer(
        storage: &HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
        inout: &Inout,
        purpose: &str,
    ) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&inout.junction) as usize;
        match storage.get(&key) {
            Some(data) => data.clone(),
            None => panic!("No {} for {:?}", purpose, inout),
        }
    }

    fn acquire_inout_buffer(
        storage: &mut HashMap<usize, Rc<RefCell<SharedTensor<f32>>>>,
        inout: &Inout,
        batch_size: usize,
        purpose: &str,
    ) -> Rc<RefCell<SharedTensor<f32>>> {
        let key = Rc::as_ptr(&inout.junction) as usize;
        match storage.get(&key) {
            Some(data) => return data.clone(),
            None => (),
        };

        let shape: TensorDesc = iter::once(batch_size)
            .chain(inout.junction.unit_shape.iter().map(|i| *i))
            .collect();
        println!("Creating {} for {:?} with shape {:?}", purpose, inout, shape);
        let buffer = SharedTensor::<f32>::new(&shape);
        let buffer_rc = Rc::new(RefCell::new(buffer));
        storage.insert(key, buffer_rc.clone());
        buffer_rc
    }
}
