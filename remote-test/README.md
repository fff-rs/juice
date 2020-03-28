# Remote Tests

Testing remotely with concourse simplifies a lot of dependencies,
such as having a certain brand of GPU in you system.

The `test.yml` is one outline of a test spec that can be run
remotely on one of the builders using the cli client to concourse,
namely [fly](https://concourse-ci.org/fly.html). Be aware that the capacity
is limited and further load might slow down other CI ops.

Act responsibly.

Fly can be obtained from [ci.spearow.io at the bottom right](https://ci.spearow.io).

## Access

Access is granted only to very few mortal souls part of the `crashtesters` `dev` sub-team.

## HowTo

For those enabled, please use from a `juice` git checkout in this sub directory:

```sh
fly -t spearow login -n crashtesters --concourse-url https://ci.spearow.io # make sure you are logged in
fly -t spearow execute -c ./test.yml --tag framework:cuda --input juice=.. --inputs-from juice-crashtest/crashtest
```

For this to work please keep a few things in mind:

* be logged in under the right team
* upload is capped at 100MB, so get rid of the `target` directories (i.e. with `rm -r $(fd -t d -I -H '^target$')` )
* experimental! unavailability might occur frequently
* do not create artifacts

Full details on how to tweak `test.yml`, checkout [concourse docs regarding running one-off tasks](https://concourse-ci.org/tasks.html#running-tasks)

### Sample output

```log
fly -t spearow execute -c ./test.yml --tag framework:cuda --input juice=. --inputs-from juice-crashtest/crashtest
uploading juice done
executing build 5210 at https://ci.spearow.io/builds/5210 
initializing
waiting for docker to come up...
Pulling quay.io/spearow/machine-learning-container-fedora-cuda@sha256:98fa1b31f5df684e1bf9c2498c7a75a7f17059744b14810519d8a4c3b143ac73...
sha256:98fa1b31f5df684e1bf9c2498c7a75a7f17059744b14810519d8a4c3b143ac73: Pulling from spearow/machine-learning-container-fedora-cuda
5c1b9e8d7bf7: Pulling fs layer
ef076e2634fa: Pulling fs layer
057ad4694a81: Pulling fs layer
2e105860db31: Pulling fs layer
5422cd9319ad: Pulling fs layer
381a9dc2184d: Pulling fs layer
fb6cef652dc2: Pulling fs layer
872b5caebd3d: Pulling fs layer
b130ffe27877: Pulling fs layer
5beec0a03be6: Pulling fs layer
fb9e8d134104: Pulling fs layer
b130ffe27877: Waiting
5beec0a03be6: Waiting

...

b130ffe27877: Download complete
5beec0a03be6: Verifying Checksum
5beec0a03be6: Download complete
fb9e8d134104: Download complete
5c1b9e8d7bf7: Pull complete
2e105860db31: Verifying Checksum
2e105860db31: Download complete
057ad4694a81: Verifying Checksum
ef076e2634fa: Download complete
ef076e2634fa: Pull complete
057ad4694a81: Pull complete
2e105860db31: Pull complete
5422cd9319ad: Pull complete
381a9dc2184d: Pull complete
fb6cef652dc2: Verifying Checksum
fb6cef652dc2: Download complete
```

that part will only appear the first time a container is required.

```log
running sh -exc prepare
cargo-override-injection
cargo build --tests --no-default-features --features=native,cuda
cargo test --no-default-features --features=native,cuda

+ prepare
Sat Mar 28 20:34:42 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.64       Driver Version: 440.64       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 770     On   | 00000000:0A:00.0 N/A |                  N/A |
| 45%   38C    P8    N/A /  N/A |    194MiB /  1996MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |

...

test tests::softmax::cuda::log_softmax_f32 ... ok
test tests::softmax::cuda::log_softmax_f64 ... ok
test tests::softmax::cuda::log_softmax_grad_f32 ... ok
test tests::softmax::cuda::log_softmax_grad_f64 ... ok
test tests::softmax::cuda::softmax_f32 ... ok
test tests::softmax::cuda::softmax_f64 ... ok
test tests::softmax::cuda::softmax_grad_f32 ... ok
test tests::softmax::cuda::softmax_grad_f64 ... ok
test tests::activation::cuda::relu_f32 ... ok

test result: ok. 89 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

   Doc-tests coaster-nn

running 1 test
test src/lib.rs -  (line 25) ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

succeeded
```