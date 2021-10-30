extern crate coaster as co;
extern crate juice;

#[cfg(test)]
mod layer_spec {
    use crate::co::prelude::*;
    use juice::layer::*;
    use juice::weight::{DimCheckMode, WeightConfig};
    use std::rc::Rc;

    // only used by cuda right now
    #[allow(dead_code)]
    fn new_layer_config() -> LayerConfig {
        LayerConfig::new("foo", LayerType::Sigmoid)
    }

    fn native_backend() -> Rc<Backend<Native>> {
        Rc::new(Backend::<Native>::default().unwrap())
    }

    #[cfg(feature = "cuda")]
    fn cuda_backend() -> Rc<Backend<Cuda>> {
        let framework = Cuda::new();
        let hardwares = framework.hardwares()[0..1].to_vec();
        let backend_config = BackendConfig::new(framework, &hardwares);
        let mut backend = Backend::new(backend_config).unwrap();
        backend.framework.initialise_cublas().unwrap();
        backend.framework.initialise_cudnn().unwrap();
        Rc::new(backend)
    }

    #[cfg(all(feature = "native", feature = "cuda"))]
    mod native_cuda {
        use super::{cuda_backend, native_backend};
        use juice::layer::*;

        #[test]
        fn create_layer_with_either() {
            let cfg = super::new_layer_config();
            Layer::from_config(native_backend(), &cfg);

            let cfg = super::new_layer_config();
            Layer::from_config(cuda_backend(), &cfg);
        }
    }

    #[cfg(feature = "native")]
    mod native {
        use super::native_backend;
        use crate::co::prelude::*;
        use juice::layer::*;
        use juice::layers::*;

        fn simple_network() -> LayerConfig {
            let mut net_cfg = SequentialConfig::default();
            net_cfg.add_input("data", &vec![1, 1, 28, 28]);
            net_cfg.add_layer(LayerConfig::new(
                "linear",
                LayerType::Linear(LinearConfig { output_size: 10 }),
            ));

            LayerConfig::new("network", net_cfg)
        }

        #[test]
        fn xor_forward() {
            // let _ = env_logger::init();
            let mut cfg = SequentialConfig::default();
            // Layer: data
            cfg.add_input("data", &[2]);
            // Layer: fc1
            cfg.add_layer(LayerConfig::new(
                "fc1",
                LayerType::Linear(LinearConfig { output_size: 2 }),
            ));
            cfg.add_layer(LayerConfig::new("fc1_out/sigmoid", LayerType::Sigmoid));
            // Layer: fc2 equiv. output
            cfg.add_layer(LayerConfig::new(
                "fc2",
                LayerType::Linear(LinearConfig { output_size: 1 }),
            ));
            cfg.add_layer(LayerConfig::new("fc2_out/sigmoid", LayerType::Sigmoid));

            let backend = native_backend();
            let _ = Layer::from_config(
                backend.clone(),
                &LayerConfig::new("network", LayerType::Sequential(cfg)),
            );
        }

        #[test]
        fn save_and_load_layer() {
            let cfg = simple_network();
            let mut original_layer = Layer::from_config(native_backend(), &cfg);
            let mut tmpfile = std::env::temp_dir();
            tmpfile.push("tmpnet");

            original_layer.save(&tmpfile).unwrap();
            let loaded_layer = Layer::<Backend<Native>>::load(native_backend(), &tmpfile).unwrap();

            assert_eq!(original_layer.input_blob_names(), loaded_layer.input_blob_names());

            let original_weights = original_layer.learnable_weights_data();
            let original_weight_lock = original_weights[0].read().unwrap();
            let loaded_weights = loaded_layer.learnable_weights_data();
            let loaded_weight_lock = loaded_weights[0].read().unwrap();

            let original_weight = original_weight_lock
                .read(native_backend().device())
                .unwrap()
                .as_slice::<f32>();
            let loaded_weight = loaded_weight_lock
                .read(native_backend().device())
                .unwrap()
                .as_slice::<f32>();

            assert_eq!(original_weight, loaded_weight);
        }
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        macro_rules! assert_slice_eq {
            ($lv:expr, $rv:expr, $eps:expr, $($arg:tt)+) => {
                {
                    let eps = $eps;
                    let lv = $lv;
                    let rv = $rv;
                    for (i,(a,b)) in lv.iter().zip(rv.iter()).enumerate() {
                        let delta = if *a < *b {
                            *a - *b
                        } else if *b < *a {
                            *b - *a
                        } else {
                            continue;
                        };
                        if delta > eps {
                            ::std::panic!(r#"assertion failed: `(left == right)`
                            left: `{:?}`,
                           right: `{:?}`: since value {} at index {} did not match {} ± {}: {}"#,
                           &lv, &rv, a, i, b, $eps,
                           ::std::format_args!($($arg)+))
                        }
                    }
                    assert_eq!(lv.iter().count(), rv.iter().count(), "Left and right handside have matching prefix, but not length.");
                }
            };

            ($lv:expr, $rv:expr, $eps:expr) => {
                {
                    let eps = $eps;
                    let lv = $lv;
                    let rv = $rv;
                    for (i,(a,b)) in lv.iter().zip(rv.iter()).enumerate() {
                        let delta = if *a < *b {
                            *b - *a
                        } else if *b < *a {
                            *a - *b
                        } else {
                            continue;
                        };
                        if delta > eps {
                            ::std::panic!(r#"assertion failed: `(left == right)`
                            left: `{:?}`,
                           right: `{:?}`: since value {} at index {} did not match {} ± {}"#,
                           &lv, &rv, a, i, b, $eps);
                        }
                    }
                    assert_eq!(lv.iter().count(), rv.iter().count(), "Left and right handside have matching prefix, but not length.");
                }
            };
        }

        #[test]
        fn macro_test_assert_slice_eq() {
            assert_slice_eq!(&[0.51], &[0.52], 0.19999999);
            assert_slice_eq!(&[0.51], &[0.51], 0.00000001);
        }

        #[test]
        #[should_panic]
        fn macro_test_assert_slice_eq_not() {
            assert_slice_eq!(&[0.51], &[0.52], 0.00000001);
        }

        #[test]
        #[should_panic]
        fn macro_test_assert_slice_eq_len() {
            assert_slice_eq!(&[0.50, 0.50], &[0.50], 0.00000001);
        }

        use super::{cuda_backend, native_backend};
        use crate::co::prelude::*;
        use juice::layer::*;
        use juice::layers::*;
        use juice::util::write_to_memory;
        use std::sync::{Arc, RwLock};

        #[test]
        fn new_layer() {
            let cfg = super::new_layer_config();
            Layer::from_config(cuda_backend(), &cfg);
        }

        #[test]
        fn can_create_default_dropout_layer() {
            let model = DropoutConfig::default();
            Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Dropout(model)));
        }

        #[test]
        fn can_create_single_dropout_layer() {
            let model = DropoutConfig {
                probability: 0.5,
                seed: 0,
            };
            Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Dropout(model)));
        }

        #[test]
        fn can_create_empty_sequential_layer() {
            let model = SequentialConfig::default();
            Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Sequential(model)));
        }

        #[test]
        fn can_create_single_layer_sequential_layer() {
            let mut model = SequentialConfig::default();
            model.add_input("data", &[28, 28]);
            model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));

            Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Sequential(model)));
        }

        #[test]
        fn can_create_simple_network_sequential_layer() {
            let mut model = SequentialConfig::default();
            model.add_input("data", &[1, 784]);
            model.add_layer(LayerConfig::new("linear1", LinearConfig { output_size: 1568 }));
            model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            model.add_layer(LayerConfig::new("linear2", LinearConfig { output_size: 10 }));
            model.add_layer(LayerConfig::new(
                "dropout",
                DropoutConfig {
                    probability: 0.8,
                    seed: 0,
                },
            ));

            let _ = Layer::from_config(cuda_backend(), &LayerConfig::new("model", LayerType::Sequential(model)));
        }

        #[test]
        fn reshape_does_not_affect_output() {
            let native_backend = native_backend();
            let cuda_backend = cuda_backend();

            let mut normal_model = SequentialConfig::default();
            normal_model.add_input("data", &[3]);
            normal_model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            let mut normal_network = Layer::from_config(
                cuda_backend.clone(),
                &LayerConfig::new("normal_model", LayerType::Sequential(normal_model)),
            );

            let mut reshape_model = SequentialConfig::default();
            reshape_model.add_input("data", &[3]);
            reshape_model.add_layer(LayerConfig::new("reshape", ReshapeConfig { shape: vec![1, 1, 3] }));
            reshape_model.add_layer(LayerConfig::new("sigmoid", LayerType::Sigmoid));
            let mut reshape_network = Layer::from_config(
                cuda_backend.clone(),
                &LayerConfig::new("reshape_model", LayerType::Sequential(reshape_model)),
            );

            let input = vec![1f32, 1f32, 2f32];
            let mut normal_tensor = SharedTensor::<f32>::new(&[3]);
            let mut reshape_tensor = SharedTensor::<f32>::new(&[3]);
            write_to_memory(normal_tensor.write_only(native_backend.device()).unwrap(), &input);
            write_to_memory(reshape_tensor.write_only(native_backend.device()).unwrap(), &input);

            let normal_tensor_output = normal_network.forward(&[Arc::new(RwLock::new(normal_tensor))])[0].clone();
            let normal_tensor_output_native_ = normal_tensor_output.read().unwrap();
            let normal_tensor_output_native = normal_tensor_output_native_.read(native_backend.device()).unwrap();
            assert_slice_eq!(
                &[0.7310585786f32, 0.7310586f32, 0.880797f32],
                normal_tensor_output_native.as_slice::<f32>(),
                1e-6_f32
            );

            let reshape_tensor_output = reshape_network.forward(&[Arc::new(RwLock::new(reshape_tensor))])[0].clone();
            let reshape_tensor_output_native_ = reshape_tensor_output.read().unwrap();
            let reshape_tensor_output_native = reshape_tensor_output_native_.read(native_backend.device()).unwrap();
            assert_slice_eq!(
                &[0.7310585786f32, 0.7310586f32, 0.880797f32],
                reshape_tensor_output_native.as_slice::<f32>(),
                1e-6_f32
            );
            assert_slice_eq!(
                normal_tensor_output_native.as_slice::<f32>(),
                reshape_tensor_output_native.as_slice::<f32>(),
                1e-6_f32
            );
        }
    }

    #[test]
    fn dim_check_strict() {
        let cfg = WeightConfig {
            share_mode: DimCheckMode::Strict,
            ..WeightConfig::default()
        };
        let blob_one = SharedTensor::<f32>::new(&vec![2, 3, 3]);
        let blob_two = SharedTensor::<f32>::new(&vec![3, 2, 3]);
        let param_name = "foo".to_owned();
        let owner_name = "owner".to_owned();
        let layer_name = "layer".to_owned();

        assert!(cfg
            .check_dimensions(
                &blob_one,
                &blob_one,
                param_name.clone(),
                owner_name.clone(),
                layer_name.clone()
            )
            .is_ok());
        assert!(cfg
            .check_dimensions(
                &blob_one,
                &blob_two,
                param_name.clone(),
                owner_name.clone(),
                layer_name.clone()
            )
            .is_err());
    }

    #[test]
    fn dim_check_permissive() {
        let cfg = WeightConfig {
            share_mode: DimCheckMode::Permissive,
            ..WeightConfig::default()
        };
        let blob_one = SharedTensor::<f32>::new(&vec![2, 3, 3]);
        let blob_two = SharedTensor::<f32>::new(&vec![3, 2, 3]);
        let blob_three = SharedTensor::<f32>::new(&vec![3, 10, 3]);
        let param_name = "foo".to_owned();
        let owner_name = "owner".to_owned();
        let layer_name = "layer".to_owned();

        assert!(cfg
            .check_dimensions(
                &blob_one,
                &blob_one,
                param_name.clone(),
                owner_name.clone(),
                layer_name.clone()
            )
            .is_ok());
        assert!(cfg
            .check_dimensions(
                &blob_one,
                &blob_two,
                param_name.clone(),
                owner_name.clone(),
                layer_name.clone()
            )
            .is_ok());
        assert!(cfg
            .check_dimensions(
                &blob_one,
                &blob_three,
                param_name.clone(),
                owner_name.clone(),
                layer_name.clone()
            )
            .is_err());
        assert!(cfg
            .check_dimensions(
                &blob_two,
                &blob_three,
                param_name.clone(),
                owner_name.clone(),
                layer_name.clone()
            )
            .is_err());
    }

    use juice::layers::SequentialConfig;
    use juice::layers::NegativeLogLikelihoodConfig;

    #[test]
    fn nll_basic() {
        const BATCH_SIZE: usize = 7;
        const KLASS_COUNT: usize = 10;
        let native_backend = native_backend();
        let mut classifier_cfg = SequentialConfig::default();
        classifier_cfg.add_input("network_out", &[BATCH_SIZE, KLASS_COUNT]);
        classifier_cfg.add_input("label", &[BATCH_SIZE, 1]);
        // set up nll loss
        let nll_layer_cfg = NegativeLogLikelihoodConfig { num_classes: 10 };
        let nll_cfg = LayerConfig::new("nll", nll_layer_cfg);
        classifier_cfg.add_layer(nll_cfg);
        let mut network = Layer::from_config(
            native_backend.clone(),
            &LayerConfig::new("foo", classifier_cfg),
        );
        let labels_data = (0..(BATCH_SIZE * KLASS_COUNT))
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let desc = [BATCH_SIZE, KLASS_COUNT];
        let desc: &[usize] = &desc[..];
        let mut input = SharedTensor::<f32>::new(&desc);
        let mem = input.write_only(native_backend.device()).unwrap();
        let input_data = (0..(KLASS_COUNT * BATCH_SIZE)).into_iter().map(|x| x as f32 * 3.77).collect::<Vec<f32>>();
        let input_data = &input_data[..];
        juice::util::write_to_memory(mem, input_data);

        // each input has exactly one label
        let labels_desc = [BATCH_SIZE, 1];
        let labels_desc = &labels_desc[..];
        let mut labels = SharedTensor::<f32>::new(&labels_desc);

        // pretend they have all different classes
        let labels_data = (1..=(BATCH_SIZE * 1))
            .into_iter()
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let mem = labels.write_only(native_backend.device()).unwrap();
        juice::util::write_to_memory(mem, labels_data.as_slice());

        let input = vec![
            std::sync::Arc::new(std::sync::RwLock::new(input)),
            std::sync::Arc::new(std::sync::RwLock::new(labels)),
        ];

        let output = network.forward(input.as_slice());

        let x = output[0].read().unwrap();
        dbg!(&x);
        let out = x.read(native_backend.device()).unwrap();
        dbg!(out.as_slice::<f32>());
    }
}
