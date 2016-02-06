//! Convolves the top Blobs
//!
//! Does this convolution with a set of learnable filters, each producing one
//! feature map in the top Blob.
use std::rc::Rc;
use co::backend::IBackend;
use co::device::DeviceType;
use co::tensor::SharedTensor;
use conn;
use layer::*;
use shared_memory::*;
use util::native_backend;

#[derive(Debug, Clone)]
/// Convolution Layer
pub struct Convolution<B: conn::Convolution<f32>> {
    axis: usize,
    num_output: usize,
    filter_shape: Vec<usize>, // TODO: rename to filter?
    stride: Vec<usize>,
    padding: Vec<usize>,

    convolution_configs: Vec<Rc<B::CC>>,
}

impl<B: conn::Convolution<f32>> Convolution<B> {
    /// Create a Convolution layer from a ConvolutionConfig.
    pub fn from_config(config: &ConvolutionConfig) -> Convolution<B> {
        Convolution {
            num_output: config.num_output,

            filter_shape: config.filter_shape.clone(),
            stride: config.stride.clone(),
            padding: config.padding.clone(),

            axis: config.axis(),

            convolution_configs: vec![],
        }
    }

    /// Calculates the number of spatial dimensions for the covolution.
    fn num_spatial_axis(&self, bottom_shape: &[usize]) -> usize {
        let num_spatial_axis = bottom_shape.len() - self.axis - 1;
        if num_spatial_axis != 2 {
            panic!("unimplemented: Only 2D convolutions are supported at the moment");
        }

        num_spatial_axis
    }

    /// Retrievs the spatial dimensions for the filter based on `self.axis` and the `bottom_shape`.
    ///
    /// The spatial dimensions only make up part of the whole filter shape. The other parts are the
    /// number of input and output feature maps.
    fn spatial_filter_dims(&self, bottom_shape: &[usize]) -> Vec<usize> {
        let num_spatial_axis = self.num_spatial_axis(bottom_shape);
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
    fn stride_dims(&self, bottom_shape: &[usize]) -> Vec<usize> {
        let num_spatial_axis = self.num_spatial_axis(bottom_shape);
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
    fn padding_dims(&self, bottom_shape: &[usize]) -> Vec<usize> {
        let num_spatial_axis = self.num_spatial_axis(bottom_shape);
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

    fn create_filter(&self, device: &DeviceType, bottom_shape: &[usize]) -> SharedTensor<f32> {
        let spatial_dims = self.spatial_filter_dims(bottom_shape);
        let filter_n = self.num_output; // number of output feature maps
        let filter_c = bottom_shape[self.axis]; // number of input feature maps
        let filter_h = spatial_dims[0];
        let filter_w = spatial_dims[1];

        SharedTensor::<f32>::new(device, &vec![filter_n, filter_c, filter_h, filter_w]).unwrap()
    }

    fn calculate_covolution_output_dims(input_dims: &[usize], filter_dims: &[usize], padding: &[usize], stride: &[usize]) -> Vec<usize> {
        let mut output_dims = Vec::with_capacity(input_dims.len());
        for (i, _) in input_dims.iter().enumerate() {
            output_dims.push(((input_dims[i] + (2 * padding[i]) - filter_dims[i]) / stride[i]) + 1);
        }
        output_dims
    }

    /// Calculate top shape based on the shape of filter, padding, stride and bottom.
    fn calculate_top_shape(&self, bottom_shape: &[usize]) -> Vec<usize> {
        let filter = self.spatial_filter_dims(bottom_shape);
        let padding = self.padding_dims(bottom_shape);
        let stride = self.stride_dims(bottom_shape);
        let mut top_shape = Vec::new();
        for dim in bottom_shape[0..self.axis].to_vec().iter() {
            top_shape.push(*dim);
        }
        top_shape.push(self.num_output);
        for spatial_dim in Self::calculate_covolution_output_dims(&bottom_shape[(self.axis + 1)..], &filter, &padding, &stride) {
            top_shape.push(spatial_dim);
        }

        top_shape
    }
}

impl<B: IBackend + conn::Convolution<f32>> ILayer<B> for Convolution<B> {
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
            let mut top_data = top_data[0].write().unwrap();
            let mut top_gradient = top_gradient[0].write().unwrap();
            let bottom_shape = btm.desc();
            let top_shape = self.calculate_top_shape(bottom_shape);
            top_data.resize(&top_shape).unwrap();
            top_gradient.resize(&top_shape).unwrap();

            let device = <B as IBackend>::device(&backend);
            let mut filter = self.create_filter(device, bottom_shape);
            let stride = vec![self.stride[0] as i32, self.stride[0] as i32]; // TODO: dimension checking etc.
            let padding = vec![self.padding[0] as i32, self.padding[0] as i32]; // TODO: dimension checking etc.

            // add copy on native as workaround for bug in new_convolution_config
            let native = native_backend();
            let _ = filter.add_device(native.device());
            let config = backend.new_convolution_config(&btm, &top_data, &mut filter,
                                                        conn::ConvForwardAlgo::Auto, conn::ConvBackwardFilterAlgo::Auto, conn::ConvBackwardDataAlgo::Auto,
                                                        &stride, &padding).unwrap();
            weights_data[0].write().unwrap().resize(filter.desc()).unwrap();
            weights_gradient[0].write().unwrap().resize(filter.desc()).unwrap();
            self.convolution_configs.push(Rc::new(config));
        }
    }
}

impl<B: IBackend + conn::Convolution<f32>> ComputeOutput<f32, B> for Convolution<B> {
    fn compute_output(&self,
                      backend: &B,
                      weights: &[&SharedTensor<f32>],
                      input_data: &[&SharedTensor<f32>],
                      output_data: &mut [&mut SharedTensor<f32>]) {
        let filter_data = weights[0];
        let input_data_i = input_data[0];
        let ref mut output_data_i = output_data[0];
        let ref conv_config = self.convolution_configs[0];
        backend.convolution_plain(filter_data, input_data_i, output_data_i, conv_config).unwrap();
    }
}

impl<B: IBackend + conn::Convolution<f32>> ComputeInputGradient<f32, B> for Convolution<B> {
    fn compute_input_gradient(&self,
                              backend: &B,
                              weights: &[&SharedTensor<f32>],
                              _output_data: &[&SharedTensor<f32>],
                              output_gradients: &[&SharedTensor<f32>],
                              input_data: &[&SharedTensor<f32>],
                              input_gradients: &mut [&mut SharedTensor<f32>]) {
        let output_gradient = output_gradients[0];
        let filter_data = weights[0];
        let ref mut input_gradient = input_gradients[0];
        let ref conv_config = self.convolution_configs[0];
        // compute gradient w.r.t. input
        backend.convolution_grad_data_plain(filter_data, output_gradient, input_gradient, conv_config).unwrap();
    }
}

impl<B: IBackend + conn::Convolution<f32>> ComputeParametersGradient<f32, B> for Convolution<B> {
    fn compute_parameters_gradient(&self,
                                   backend: &B,
                                   _output_data: &[&SharedTensor<f32>],
                                   output_gradients: &[&SharedTensor<f32>],
                                   input_data: &[&SharedTensor<f32>],
                                   parameters_gradients: &mut [&mut SharedTensor<f32>]) {
        // TODO: compute gradient w.r.t to bias
        let input_data_i = input_data[0];
        let output_gradient = output_gradients[0];
        let ref mut filter_gradient = parameters_gradients[0];
        let ref conv_config = self.convolution_configs[0];
        // compute gradient w.r.t. filter
        backend.convolution_grad_filter_plain(input_data_i, output_gradient, filter_gradient, conv_config).unwrap();
    }
}


#[derive(Debug, Clone)]
/// Specifies configuration parameters for a Convolution Layer.
pub struct ConvolutionConfig {
    /// The number of output values
    pub num_output: usize,
    /// The size of the kernel
    pub filter_shape: Vec<usize>,
    /// The stride size
    pub stride: Vec<usize>,
    /// The padding size
    pub padding: Vec<usize>,
    /// The axis to interpret as "channels" when performing convolution.
    ///
    /// Preceding dimensions are treated as independent inputs, and
    /// succeeding dimensions are treated as "spatial".
    ///
    /// Defaults to `1`
    pub axis: Option<usize>,
}

impl ConvolutionConfig {
    /// The axis to interpret as "channels" when performing convolution.
    ///
    /// Preceding dimensions are treated as independent inputs, and
    /// succeeding dimensions are treated as "spatial".
    ///
    /// Defaults to `1`
    pub fn axis(&self) -> usize {
        self.axis.unwrap_or(1)
    }
}
