//! Applies pooling to the top blobs.
use std::rc::Rc;
use co::IBackend;
use co::SharedTensor;
use conn;
use layer::*;
use shared_memory::*;
use util::cast_vec_usize_to_i32;

#[derive(Debug, Clone)]
/// Pooling Layer
pub struct Pooling<B: conn::Pooling<f32>> {
    mode: PoolingMode,

    axis: usize,

    filter_shape: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,

    pooling_configs: Vec<Rc<B::CPOOL>>,
}

impl<B: conn::Pooling<f32>> Pooling<B> {
    /// Create a Pooling layer from a PoolingConfig.
    pub fn from_config(config: &PoolingConfig) -> Pooling<B> {
        Pooling {
            mode: config.mode,

            axis: config.axis(),

            filter_shape: config.filter_shape.clone(),
            stride: config.stride.clone(),
            padding: config.padding.clone(),

            pooling_configs: vec![],
        }
    }

    /// Calculates the number of spatial dimensions for the pooling operation.
    fn num_spatial_axis(&self, bottom_shape: &[usize]) -> usize {
        bottom_shape.len() - self.axis - 1
    }

    /// Retrievs the spatial dimensions for the filter based on `self.axis` and the `bottom_shape`.
    ///
    /// The spatial dimensions only make up part of the whole filter shape. The other parts are the
    /// number of input and output feature maps.
    fn spatial_filter_dims(&self, num_spatial_axis: usize) -> Vec<usize> {
        let mut spatial_dims = Vec::with_capacity(num_spatial_axis);
        if self.filter_shape.len() == 1 {
            for i in 0..num_spatial_axis {
                spatial_dims.push(self.filter_shape[0]);
            }
        } else if self.filter_shape.len() == num_spatial_axis {
            panic!("unimplemented: You can not yet specify one filter dimension per spatial axis");
        } else {
            panic!("Must either specify one filter_shape or one filter_shape per spatial axis. Supplied {:?}", self.filter_shape.len());
        }

        spatial_dims
    }

    /// Retrievs the stride for the convolution based on `self.axis` and the `bottom_shape`
    fn stride_dims(&self, num_spatial_axis: usize) -> Vec<usize> {
        let mut stride_dims = Vec::with_capacity(num_spatial_axis);
        if self.stride.len() == 1 {
            for i in 0..num_spatial_axis {
                stride_dims.push(self.stride[0]);
            }
        } else if self.stride.len() == num_spatial_axis {
            panic!("unimplemented: You can not yet specify one stride per spatial axis");
        } else {
            panic!("Must either specify one stride or one stride per spatial axis. Supplied {:?}", self.stride.len());
        }

        stride_dims
    }

    /// Retrievs the padding for the convolution based on `self.axis` and the `bottom_shape`
    fn padding_dims(&self, num_spatial_axis: usize) -> Vec<usize> {
        let mut padding_dims = Vec::with_capacity(num_spatial_axis);
        if self.padding.len() == 1 {
            for i in 0..num_spatial_axis {
                padding_dims.push(self.padding[0]);
            }
        } else if self.padding.len() == num_spatial_axis {
            panic!("unimplemented: You can not yet specify one padding per spatial axis");
        } else {
            panic!("Must either specify one padding or one padding per spatial axis. Supplied {:?}", self.padding.len());
        }

        padding_dims
    }

    fn calculate_covolution_output_dims(input_dims: &[usize], filter_dims: &[usize], padding: &[usize], stride: &[usize]) -> Vec<usize> {
        let mut output_dims = Vec::with_capacity(input_dims.len());
        for (i, _) in input_dims.iter().enumerate() {
            output_dims.push(((input_dims[i] + (2 * padding[i]) - filter_dims[i]) / stride[i]) + 1);
        }
        output_dims
    }

    fn calculate_top_shape(&self, bottom_shape: &[usize]) -> Vec<usize> {
        let num_spatial_axis = self.num_spatial_axis(bottom_shape);
        let filter = self.spatial_filter_dims(num_spatial_axis);
        let padding = self.padding_dims(num_spatial_axis);
        let stride = self.stride_dims(num_spatial_axis);
        let mut top_shape = Vec::new();
        for dim in bottom_shape[0..self.axis + 1].to_vec().iter() {
            top_shape.push(*dim);
        }
        for spatial_dim in Self::calculate_covolution_output_dims(&bottom_shape[(self.axis + 1)..], &filter, &padding, &stride) {
            top_shape.push(spatial_dim);
        }

        top_shape
    }
}

impl<B: IBackend + conn::Pooling<f32>> ILayer<B> for Pooling<B> {
    impl_ilayer_common!();

    fn reshape(&mut self,
               backend: ::std::rc::Rc<B>,
               bottom_data: &[ArcLock<SharedTensor<f32>>],
               weights_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               weights_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_data: &mut Vec<ArcLock<SharedTensor<f32>>>,
               top_gradient: &mut Vec<ArcLock<SharedTensor<f32>>>) {
        for i in 0..bottom_data.len() {
            let btm = bottom_data[0].read().unwrap();
            let bottom_shape = btm.desc();
            let top_shape = self.calculate_top_shape(bottom_shape);
            top_data[0].write().unwrap().resize(&top_shape).unwrap();
            top_gradient[0].write().unwrap().resize(&top_shape).unwrap();

            let num_spatial_axis = self.num_spatial_axis(btm.desc());
            let filter = cast_vec_usize_to_i32(self.spatial_filter_dims(num_spatial_axis));
            let stride = cast_vec_usize_to_i32(self.stride_dims(num_spatial_axis));
            let padding = cast_vec_usize_to_i32(self.padding_dims(num_spatial_axis));

            let config = backend.new_pooling_config(&filter, &padding, &stride).unwrap();
            self.pooling_configs.push(Rc::new(config));
        }
    }
}

impl<B: IBackend + conn::Pooling<f32>> ComputeOutput<f32, B> for Pooling<B> {
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        // let input_data_i = input_data[0];
        // let ref mut output_data_i = output_data[0];

        let ref config = self.pooling_configs[0];
        match self.mode {
            PoolingMode::Max => backend.pooling_max_plain(input_data[0], output_data[0], &*config).unwrap(),
            // TODO: implement average pooling
            PoolingMode::Average => unimplemented!(),
        }
    }
}

impl<B: IBackend + conn::Pooling<f32>> ComputeInputGradient<f32, B> for Pooling<B> {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights: &[&SharedTensor<f32>],
                              output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        let ref config = self.pooling_configs[0];
        backend.pooling_max_grad_plain(output_data[0], output_gradients[0], input_data[0], input_gradients[0], &*config).unwrap();
    }
}

impl<B: IBackend + conn::Pooling<f32>> ComputeParametersGradient<f32, B> for Pooling<B> {
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   output_data: &[&SharedTensor<f32>],
                                   output_gradients: &[&SharedTensor<f32>],
                                   input_data: &[&SharedTensor<f32>],
                                   parameters_gradients: &mut [&mut SharedTensor<f32>]) {
        // // TODO: compute gradient w.r.t to bias
        // let input_data_i = input_data[0];
        // let output_gradient = output_gradients[0];
        // let ref mut filter_gradient = parameters_gradients[0];
        // let ref conv_config = self.convolution_configs[0];
        // // compute gradient w.r.t. filter
        // backend.convolution_grad_filter_plain(input_data_i, output_gradient, filter_gradient, conv_config).unwrap();
    }
}

#[derive(Debug, Clone)]
/// Specifies configuration parameters for a Pooling Layer.
pub struct PoolingConfig {
    /// The PoolingMode to use
    pub mode: PoolingMode,
    /// The shape of the filter
    pub filter_shape: Vec<usize>,
    /// The stride size
    pub stride: Vec<usize>,
    /// The padding size
    pub padding: Vec<usize>,
    /// The axis to interpret as "channels" when performing pooling.
    /// Preceding dimensions are treated as independent inputs, and
    /// succeeding dimensions are treated as "spatial".
    ///
    /// Defaults to `1`
    pub axis: Option<usize>,
}

impl PoolingConfig {
    /// The axis to interpret as "channels" when performing convolution.
    /// Preceding dimensions are treated as independent inputs, and
    /// succeeding dimensions are treated as "spatial".
    ///
    /// Defaults to `1`
    pub fn axis(&self) -> usize {
        self.axis.unwrap_or(1)
    }
}

#[derive(Debug, Copy, Clone)]
/// The different modes of pooling that can be calculates.
pub enum PoolingMode {
    /// The maximum value inside the pooling window will be used as result.
    Max,
    /// The average of all values inside the pooling window will be used as result.
    Average,
}
