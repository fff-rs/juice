<a name="0.0.7"></a>
## 0.0.7 (2015-12-21)


#### Features

* **backend_device:**  add device() to IBackend ([f797a72a](https://github.com/autumnai/collenchyma/commit/f797a72a39a530f2d29f7dc25c0b1c11ec7cda87))
* **tensor:**  add reshape to SharedTensor ([9c8721ed](https://github.com/autumnai/collenchyma/commit/9c8721edd5acc66caa955253d9fdb403600f9b85))

#### Bug Fixes

* **lib:**  various fixes concerning impl of plugins ([bec27ca1](https://github.com/autumnai/collenchyma/commit/bec27ca1de82b1be21ca2295a65d56311533ccba))



<a name="0.0.6"></a>
## 0.0.6 (2015-12-16)


#### Features

* **tensor:**  rename SharedMemory -> Tensor ([acc3cbb7](https://github.com/autumnai/collenchyma/commit/acc3cbb7f0f850cf6414131f33c013f71a53a852))

#### Bug Fixes

* **travis:**  fix feature build flags ([f9861a73](https://github.com/autumnai/collenchyma/commit/f9861a731360f3a3dc7ad6f81a69be6cc05fe622))



<a name="0.0.5"></a>
## 0.0.5 (2015-12-09)


#### Bug Fixes

* **use-size-types:**  use target dependent size types ([4e4a5cd3](https://github.com/autumnai/collenchyma/commit/4e4a5cd3f1716c122a22bb0d008b06bb61f74bce))

#### Features

* **bench:**  add benchmarks for memory synchronization ([762b87ca](https://github.com/autumnai/collenchyma/commit/762b87ca2c1bf55e257803607d7a8463c07e77e3))
* **features:**  add native and opencl feature ([3609fea1](https://github.com/autumnai/collenchyma/commit/3609fea17d1cd196196d7dba3140ede53e681d41))



<a name="0.0.4"></a>
## 0.0.4 (2015-12-08)


#### Features

* **cuda:**
  *  add memory synchronization ([760e4c45](https://github.com/autumnai/collenchyma/commit/760e4c45f97312729770e51980ad1481906d33b4))
  *  add cuda device support ([62eef6d8](https://github.com/autumnai/collenchyma/commit/62eef6d8cfb8a60f6fac892b50301738adaaa65c))
  *  add memory allocation ([35f7f479](https://github.com/autumnai/collenchyma/commit/35f7f47916cebcea626e43313c3f42ebce2e4e21))
  *  add basic context and memory support ([b0a40d38](https://github.com/autumnai/collenchyma/commit/b0a40d38f46a3ef313495a0d4d0847db0821bb64))
* **cudnn:**  add cudnn ffi ([0bbbff83](https://github.com/autumnai/collenchyma/commit/0bbbff832632c868589f8905e6e9e70d003161c2))
* **library:**  remove last pieces of library ([38dcd6a6](https://github.com/autumnai/collenchyma/commit/38dcd6a68914ab070d76a89f19307ff58beaccf7))
* **opencl:**  implement shared_memory for OpenCL ([be47d6ba](https://github.com/autumnai/collenchyma/commit/be47d6ba6a5de216726bd4adc0dd2f99fbe7c31b))
* **perf:**  make error messages static strings ([430c4ed6](https://github.com/autumnai/collenchyma/commit/430c4ed657242dce21ebafad7084e94d9755fae3))
* **plugin:**  move library out; replace with thin plugin mod ([3bbebe9a](https://github.com/autumnai/collenchyma/commit/3bbebe9a7f95ce24a936b3824c5ce410c79e7214))
* **shared_memory:**  add dimensionality to shared_memory ([13cd0905](https://github.com/autumnai/collenchyma/commit/13cd090596358e523c70e98dcf32885b2b9271bd))

#### Performance

* **blas:**  reduce overhead of calling blas ([8b7a7aee](https://github.com/autumnai/collenchyma/commit/8b7a7aeeaf67482031da0fd712328f747be09e72))
* **shared_memory:**  use linear_map for SharedMemory.copies ([44ea377d](https://github.com/autumnai/collenchyma/commit/44ea377d08da066159a01d646fed65f5f7080f8f))

#### Bug Fixes

* **compilation:**  make cuda optional ([1f933977](https://github.com/autumnai/collenchyma/commit/1f9339771d1eec9a6c42bcbed1ff784f97220896))
* **windows:**
  *  use `size_t` instead of `u64` where necessary ([6e9fdfbb](https://github.com/autumnai/collenchyma/commit/6e9fdfbb73155927a91b7b31dff2b208c3e49624))
  *  add the link attribute to OpenCL external block ([2017a10f](https://github.com/autumnai/collenchyma/commit/2017a10fcf8597b83f9f4ae11a6396927406c81d))



<a name="0.0.3"></a>
## 0.0.3 (2015-11-30)


#### Features

* **blas:**
  *  add blas native level 1 support ([38273645](https://github.com/autumnai/collenchyma/commit/3827364549dfa5b79ef2bbbb0bd38f0096e267cc))
  *  add basic level 1 blas methods for native ([62cbc4c4](https://github.com/autumnai/collenchyma/commit/62cbc4c42757a7a489a358c2bd5e16bdb47938cd))
* **computation:**  add basic design for backend-agnostic computation ([a3f9534f](https://github.com/autumnai/collenchyma/commit/a3f9534f9483531be4eecd91610d2e72ae84cd07))
* **cuda:**
  *  add cuda structure ([d42430c1](https://github.com/autumnai/collenchyma/commit/d42430c10248f39abb665a7bb22d0b3e32e2f08d))
* **dot:**  add working dot computation ([6572c010](https://github.com/autumnai/collenchyma/commit/6572c01036a4ac07906852f5bfaee3d7709d8f8c))
* **extern:**  add backend traits for extern support ([f3d50172](https://github.com/autumnai/collenchyma/commit/f3d5017228ebc4195d593d0210e88c4ca431eaa4))
* **flatbox:**  provide slice accessors and give more allocation responsibility to shared_memory ([cfbb5b13](https://github.com/autumnai/collenchyma/commit/cfbb5b135691f58671f7e391ec16793921073198))
* **memory:**
  *  implement MemoryType unwrappers ([fbd26776](https://github.com/autumnai/collenchyma/commit/fbd26776685aa6e4d90253628099104df1759871))
  *  add SharedMemory.latest_device() ([d425fc6c](https://github.com/autumnai/collenchyma/commit/d425fc6c29b177f778d8f3ed1c39fe5328191537))

#### Bug Fixes

* **ci:**  change clippy to be optional ([753dfb02](https://github.com/autumnai/collenchyma/commit/753dfb0203d2ff9e063e04bf14d93b40fbc506b5))


<a name="0.0.2"></a>
## 0.0.2 (2015-11-27)


#### Features

* **collenchyma:**  outline design ([876ac4b0](https://github.com/autumnai/collenchyma/commit/876ac4b037cda06975f1c88dfb8d86bc62d7fe47))
* **computation:**  add basic design for backend-agnostic computation ([e43f947a](https://github.com/autumnai/collenchyma/commit/e43f947adc12f02b5d5f84a9a4d4cce0d902580c))
* **dot:**  add working dot computation ([f5c8fdaa](https://github.com/autumnai/collenchyma/commit/f5c8fdaa11a43ef5b3244e2480afabe2f7374248))
* **flatbox:**  provide slice accessors and give more allocation responsibility to shared_memory ([a31dd493](https://github.com/autumnai/collenchyma/commit/a31dd4936c7bc37b8184bd8a8194e153b6826ec7))
* **lib:**  add foundation of the library design ([f5fd0235](https://github.com/autumnai/collenchyma/commit/f5fd02352e07094a457dff3b058a6842b39f798f))
* **memory:**
  *  implement MemoryType unwrappers ([a78edebb](https://github.com/autumnai/collenchyma/commit/a78edebb0036c82617847a58e5c4cb8cc5995d5b))
  *  add SharedMemory.latest_device() ([f9a7465d](https://github.com/autumnai/collenchyma/commit/f9a7465d54808672a5c39cb4cf89d8a0253e8aec))

#### Bug Fixes

* **ci:**  change clippy to be optional ([db13da29](https://github.com/autumnai/collenchyma/commit/db13da299ee635a4201ff66015a6df2e65314e73))




