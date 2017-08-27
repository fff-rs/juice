### Error Description

> error logs, as verbose as possible
> describe which command you ran


### What should have happend

> your expectations


### Describe in which context it happened

> custom app, demo compilation, test run


### Environment

OS:

> `uname -a`
> `cat /etc/os-release | grep PRETTY_NAME`

GPU Devices:

> `lspci | grep "\(VGA\|GPU\)"`
> `lsmod | grep "\(amdgpu\|radeon\|nvidia\|nouveau\|i915\)"`

Native related issues:

> `pkg-config --libs --cflags blas`

cuda related issues:

> `pkg-config --cflags cublas cudnn`
> `env | grep "CU\(BLAS\|DNN\|DA\)_.*"`

OpenCL related issues:

> `clinfo`
