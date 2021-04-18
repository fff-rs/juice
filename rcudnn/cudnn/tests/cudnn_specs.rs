extern crate coaster as co;
extern crate libc;
extern crate rcudnn as cudnn;

extern crate rcudnn_sys as ffi;
use crate::ffi::*;

#[cfg(test)]
#[link(name = "cudart")]
#[cfg(test)]
mod cudnn_spec {

    use rcudnn::{cudnnRNNDescriptor_t, cudnnRNNPaddingMode_t, cudnnTensorDescriptor_t};

    use crate::co::framework::IFramework;

    use crate::co::frameworks::Cuda;
    use crate::cudnn::cuda::CudaDeviceMemory;
    use crate::cudnn::utils::DataType;
    use crate::cudnn::utils::DropoutConfig;
    use crate::cudnn::utils::RnnConfig;
    use crate::cudnn::{
        ActivationDescriptor, ConvolutionDescriptor, Cudnn, FilterDescriptor, TensorDescriptor, API, tensor_vec_id_c,
    };

    #[test]
    fn it_initializes_correctly() {
        let cuda = Cuda::new();
        println!("{:?}", cuda.hardwares());
        match Cudnn::new() {
            Ok(_) => assert!(true),
            Err(err) => {
                println!("{:?}", err);
                assert!(false);
            }
        }
    }

    #[test]
    fn it_returns_version() {
        println!("cuDNN Version: {:?}", Cudnn::version());
    }

    /*
     * Results sometimes in weird memory allocation problems, with other tests if we run this test.
     * Might be due to the strange and totally not actually working memory pointers
     * `unsafe { transmute::<u64, *const ::libc::c_void>(1u64) }`.
     *
     * Since then this has been rewritten to not use transmute but a sequence of unsafe optimizations.
     */
    #[test]
    fn it_computes_sigmoid() {
        let cudnn = Cudnn::new().unwrap();
        let desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        let acti =
            ActivationDescriptor::new(crate::cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID)
                .unwrap();

        let mut a: u64 = 1;
        let a_ptr: *mut u64 = &mut a;
        let mut b: u64 = 1;
        let b_ptr: *mut u64 = &mut b;
        unsafe {
            let mut x: *mut ::libc::c_void = ::std::ptr::null_mut();
            crate::cudaHostAlloc(&mut x, 2 * 2 * 2, 0);
            match API::activation_forward(
                *cudnn.id_c(),
                *acti.id_c(),
                a_ptr as *mut ::libc::c_void,
                *desc.id_c(),
                x,
                b_ptr as *mut ::libc::c_void,
                *desc.id_c(),
                x,
            ) {
                Ok(_) => assert!(true),
                Err(err) => {
                    println!("{:?}", err);
                    assert!(false)
                }
            }
            crate::cudaFreeHost(x);
        }
    }

    #[test]
    fn it_finds_correct_convolution_algorithm_forward() {
        let cudnn = Cudnn::new().unwrap();
        let src = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        let filter = FilterDescriptor::new(&[1, 1, 1], DataType::Float).unwrap();
        let conv = ConvolutionDescriptor::new(&[1, 1, 1], &[1, 1, 1], DataType::Float).unwrap();
        let dest = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        match API::find_convolution_forward_algorithm(
            *cudnn.id_c(),
            *filter.id_c(),
            *conv.id_c(),
            *src.id_c(),
            *dest.id_c(),
        ) {
            Ok(algos) => assert_eq!(2, algos.len()),
            Err(err) => {
                println!("{:?}", err);
                assert!(false)
            }
        }
    }

    #[test]
    fn it_finds_correct_convolution_algorithm_backward() {
        let cudnn = Cudnn::new().unwrap();
        let src = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        let filter = FilterDescriptor::new(&[1, 1, 1], DataType::Float).unwrap();
        let conv = ConvolutionDescriptor::new(&[1, 1, 1], &[1, 1, 1], DataType::Float).unwrap();
        let dest = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
        match API::find_convolution_backward_data_algorithm(
            *cudnn.id_c(),
            *filter.id_c(),
            *conv.id_c(),
            *src.id_c(),
            *dest.id_c(),
        ) {
            Ok(algos) => assert_eq!(2, algos.len()),
            Err(err) => {
                println!("{:?}", err);
                assert!(false)
            }
        }
    }

    #[test]
    fn it_allocates_cuda_device_memory() {
        let _ = Cudnn::new().unwrap();
        let _ = CudaDeviceMemory::new(1024).unwrap();
    }

    #[test]
    fn it_computes_dropout_forward() {
        let cudnn = Cudnn::new().unwrap();
        let src = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();

        let result = cudnn.init_dropout(0.5, 27);
        let cfg: DropoutConfig = result.unwrap();
        let ref drop = cfg.dropout_desc();
        let ref reserve = cfg.reserved_space();
        let dest = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();

        let src_data = CudaDeviceMemory::new(2 * 2 * 2 * 4).unwrap();
        let dest_data = CudaDeviceMemory::new(2 * 2 * 2 * 4).unwrap();

        API::dropout_forward(
            *cudnn.id_c(),
            *drop.id_c(),
            *src.id_c(),
            *src_data.id_c(),
            *dest.id_c(),
            *dest_data.id_c(),
            *reserve.id_c(),
            reserve.size(),
        ).expect("dropout_forward works. qed");
    }




    #[test]
    fn it_computes_rnn_forward_backward_data() {
        use rcudnn::utils::DataTypeInfo;
        use rcudnn::RnnDescriptor;

        let cudnn = Cudnn::new().unwrap();

        let cfg: DropoutConfig = cudnn.init_dropout(0.5, 27).unwrap();
        let ref dropout_desc = cfg.dropout_desc();

        let forward_mode = rcudnn::cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING;
        let direction_mode = rcudnn::cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL;
        let bidirectional = if direction_mode == rcudnn::cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL {
            1
        } else {
            2 // bidirection needs twice as much memory
        };
        let network_mode = rcudnn::cudnnRNNMode_t::CUDNN_LSTM;
        let algorithm = rcudnn::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD;
        let input_mode = rcudnn::cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT;
        let sequence_length = 7;
        let hidden_size = 5;
        let num_layers = 3;
        let batch_size = 1;
        let data_type = DataType::Float;
        let input_size = 11;

        let mut x_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
        let mut y_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
        let mut dx_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);
        let mut dy_desc: Vec<TensorDescriptor> = Vec::with_capacity(sequence_length as usize);


        // Treating the input split by batch then input like in a typical NCHW cell.
        let dim_input = vec![batch_size, input_size, 1];
        let dim_output = vec![batch_size, hidden_size * hidden_size, 1];
        let dim_hidden_cell = vec![num_layers, batch_size, hidden_size];
        let stride_input = vec![dim_input[2] * dim_input[1], dim_input[2], 1];
        let stride_output = vec![dim_output[2] * dim_output[1], dim_output[2], 1];
        let stride_hidden_cell = vec![
            dim_hidden_cell[2] * dim_hidden_cell[1],
            dim_hidden_cell[2],
            1,
        ];

        let mut rnn_desc = API::create_rnn_descriptor().unwrap();
        API::set_rnn_descriptor(
            *cudnn.id_c(),
            rnn_desc,
            hidden_size,
            num_layers,
            *dropout_desc.id_c(),
            input_mode,
            direction_mode,
            network_mode,
            algorithm,
            data_type,
        ).unwrap();


        let weights_size = API::get_rnn_params_size(*cudnn.id_c(),
            rnn_desc,
            *x_desc[0].id_c(),
            data_type).unwrap() as i32;
        let filter_dims = vec![weights_size / std::mem::size_of::<f32>() as i32, 1_i32, 1];
        let w_desc = FilterDescriptor::new(&filter_dims, data_type).unwrap();
        let w = CudaDeviceMemory::new(weights_size as usize).unwrap();
        let dw_desc = FilterDescriptor::new(&filter_dims, data_type).unwrap();
        let dw = CudaDeviceMemory::new(weights_size as usize).unwrap();

        let size_x_dx = (sequence_length * input_size * batch_size) as usize * std::mem::size_of::<f32>();
        let size_y_dy = (sequence_length * hidden_size * batch_size * bidirectional) as usize * std::mem::size_of::<f32>();

        let size_the_rest = (num_layers * hidden_size * batch_size * bidirectional) as usize * std::mem::size_of::<f32>();


        let x = CudaDeviceMemory::new(size_x_dx).unwrap();
        let y = CudaDeviceMemory::new(size_y_dy).unwrap();

        let dx = CudaDeviceMemory::new(size_x_dx).unwrap();
        let dy = CudaDeviceMemory::new(size_y_dy).unwrap();

        let hx_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let hx = CudaDeviceMemory::new(size_the_rest).unwrap();

        let hy_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let hy = CudaDeviceMemory::new(size_the_rest).unwrap();

        let cx_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let cx = CudaDeviceMemory::new(size_the_rest).unwrap();

        let cy_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let cy = CudaDeviceMemory::new(size_the_rest).unwrap();

        let dhx_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let dhx = CudaDeviceMemory::new(size_the_rest).unwrap();

        let dhy_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let dhy = CudaDeviceMemory::new(size_the_rest).unwrap();

        let dcx_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let dcx = CudaDeviceMemory::new(size_the_rest).unwrap();

        let dcy_desc = TensorDescriptor::new(&dim_hidden_cell, &stride_hidden_cell, data_type).unwrap();
        let dcy = CudaDeviceMemory::new(size_the_rest).unwrap();


        let (workspace_size, reserved_size) = API::get_rnn_temp_space_size(
            *cudnn.id_c(),
            rnn_desc,
            forward_mode,
            x_desc.iter().map(|x| *x.id_c() ).collect::<Vec<cudnnTensorDescriptor_t>>(),
        ).unwrap();


        for _ in 0..sequence_length {
            x_desc.push(TensorDescriptor::new(&dim_input, &stride_input, data_type).unwrap());
            dx_desc.push(TensorDescriptor::new(&dim_input, &stride_input, data_type).unwrap());
            y_desc.push(TensorDescriptor::new(&dim_output, &stride_output, data_type).unwrap());
            dy_desc.push(TensorDescriptor::new(&dim_output, &stride_output, data_type).unwrap());
        }

        // assert_eq!(workspace_size, rnn.workspace_size());
        // assert_eq!(reserved_size, rnn.training_reserve_size());

        let workspace = CudaDeviceMemory::new(workspace_size).unwrap();
        let reserved = CudaDeviceMemory::new(reserved_size).unwrap();


        API::rnn_forward_training(
            *cudnn.id_c(),
            rnn_desc,
            sequence_length,
            tensor_vec_id_c(&x_desc),
            *x.id_c(),
            *hx_desc.id_c(),
            *hx.id_c(),
            *cx_desc.id_c(),
            *cx.id_c(),
            *w_desc.id_c(),
            *w.id_c(),
            tensor_vec_id_c(&y_desc),
            *y.id_c(),
            *hy_desc.id_c(),
            *hy.id_c(),
            *cy_desc.id_c(),
            *cy.id_c(),
            *workspace.id_c(),
            workspace_size,
            *reserved.id_c(),
            reserved_size,
        ).expect("API::rnn_forward_training works");


        API::rnn_backward_data(*cudnn.id_c(),
            rnn_desc,
            sequence_length,
            y_desc[0].id_c(),
            *y.id_c(),
            dy_desc[0].id_c(),
            *dy.id_c(),
            *dhy_desc.id_c(),
            *dhy.id_c(),
            *dcy_desc.id_c(),
            *dcy.id_c(),
            *w_desc.id_c(),
            *w.id_c(),
            *hx_desc.id_c(),
            *hx.id_c(),
            *cx_desc.id_c(),
            *cx.id_c(),
            dx_desc[0].id_c(),
            *dx.id_c(),
            *dhx_desc.id_c(),
            *dhx.id_c(),
            *dcx_desc.id_c(),
            *dcx.id_c(),
            *workspace.id_c(),
            workspace_size,
            *reserved.id_c(),
            reserved_size).unwrap();


        API::rnn_backward_weights(*cudnn.id_c(),
            rnn_desc,
            sequence_length,
            x_desc[0].id_c(),
            *x.id_c(),
            *hx_desc.id_c(),
            *hx.id_c(),
            y_desc[0].id_c(),
            *y.id_c(),
            *workspace.id_c(),
            workspace_size,
            *dw_desc.id_c(),
            *dw.id_c(),
            *reserved.id_c(),
            reserved_size
        ).unwrap();
        // cudnn.rnn_forward(rnn_config,
        //     src_desc,
        //     src,
        //     output_desc,
        //     output,
        //     hidden_desc,
        //     hidden,
        //     cell_desc,
        //     cell,
        //     weight_desc,
        //     weight,
        //     hidden_output_desc,
        //     hidden_output,
        //     cell_output_desc,
        //     cell_output,
        //     workspace,
        //     reserve_data)
        //     .expect("rnn_forward works");
    }
}
