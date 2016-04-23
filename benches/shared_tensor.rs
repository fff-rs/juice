#![feature(test)]
#![feature(clone_from_slice)]

extern crate test;
extern crate collenchyma as co;
extern crate rand;

use test::Bencher;
use co::device::{IDevice, DeviceType};
use co::backend::{Backend, BackendConfig};
#[cfg(feature = "native")]
use co::frameworks::Native;
#[cfg(feature = "native")]
use co::frameworks::native;
#[cfg(feature = "opencl")]
use co::frameworks::OpenCL;
#[cfg(feature = "opencl")]
use co::frameworks::opencl;
#[cfg(feature = "cuda")]
use co::frameworks::Cuda;
#[cfg(feature = "cuda")]
use co::frameworks::cuda;
use co::framework::IFramework;
use co::tensor::SharedTensor;

use rand::{thread_rng, Rng};

#[cfg(feature = "native")]
fn native_backend() -> Backend<Native> {
    let framework = Native::new();
    let hardwares = framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, &hardwares);
    Backend::new(backend_config).unwrap()
}

#[cfg(feature = "cuda")]
fn cuda_backend() -> Backend<Cuda> {
    let framework = Cuda::new();
    let hardwares = framework.hardwares()[0..1].to_vec();
    let backend_config = BackendConfig::new(framework, &hardwares);
    Backend::new(backend_config).unwrap()
}

#[cfg(feature = "opencl")]
fn opencl_backend() -> Backend<OpenCL> {
    let framework = OpenCL::new();
    let hardwares = framework.hardwares()[1..2].to_vec();
    let backend_config = BackendConfig::new(framework, &hardwares);
    Backend::new(backend_config).unwrap()
}

#[inline(never)]
fn sync_back_and_forth(
    b: &mut Bencher,
    n: usize,
    nt_device: &DeviceType,
    cl_device: &DeviceType,
    mem: &mut SharedTensor<u8>
) {
    b.iter(|| {
        for _ in 0..n {
            match mem.read_write(&cl_device) {
                Ok(_) => assert!(true),
                Err(err) => {
                    println!("{:?}", err);
                    assert!(false);
                }
            }
            match mem.read_write(&nt_device) {
                Ok(_) => assert!(true),
                Err(err) => {
                    println!("{:?}", err);
                    assert!(false);
                }
            }
        }
    });
}

#[bench]
#[cfg(feature = "opencl")]
fn bench_256_alloc_1mb_opencl(b: &mut Bencher) {
    let opencl_backend = opencl_backend();
    if let &DeviceType::OpenCL(ref cl_device) = opencl_backend.device() {
        bench_256_alloc_1mb_opencl_profile(b, cl_device, 1_048_576);
    } else {
        assert!(false);
    }
}

#[inline(never)]
#[cfg(feature = "opencl")]
fn bench_256_alloc_1mb_opencl_profile(
    b: &mut Bencher,
    device: &opencl::Context,
    size: usize
) {
    b.iter(|| {
        for _ in 0..256 {
            device.alloc_memory(size);
        }
    });
}

#[bench]
#[cfg(feature = "opencl")]
fn bench_256_sync_1mb_native_opencl(b: &mut Bencher) {
    let opencl_backend = opencl_backend();
    let cl_device = opencl_backend.device();
    let native_backend = native_backend();
    let nt_device = native_backend.device();
    // if let &DeviceType::OpenCL(ref cl_d) = cl_device {
    //     println!("{:?}", cl_d.hardwares()[0].clone().load_name());
    // }
    let mem = &mut SharedTensor::<u8>::new(&1_048_576);
    mem.write_only(&cl_device);
    bench_256_sync_1mb_native_opencl_profile(b, nt_device, cl_device, mem);
}

#[inline(never)]
#[cfg(feature = "opencl")]
fn bench_256_sync_1mb_native_opencl_profile(b: &mut Bencher, nt_device: &DeviceType, cl_device: &DeviceType, mem: &mut SharedTensor<u8>) {
    sync_back_and_forth(b, 256, nt_device, cl_device, mem)
}

#[bench]
#[cfg(feature = "cuda")]
fn bench_256_sync_1mb_native_cuda(b: &mut Bencher) {
    let cuda_backend = cuda_backend();
    let cl_device = cuda_backend.device();
    let native_backend = native_backend();
    let nt_device = native_backend.device();
    // if let &DeviceType::Cuda(ref cl_d) = cl_device {
    //     println!("{:?}", cl_d.hardwares()[0].clone().load_name());
    // }
    let mem = &mut SharedTensor::<u8>::new(&1_048_576);
    mem.write_only(&cl_device);
    bench_256_sync_1mb_native_cuda_profile(b, nt_device, cl_device, mem);
}

#[inline(never)]
#[cfg(feature = "cuda")]
fn bench_256_sync_1mb_native_cuda_profile(b: &mut Bencher, nt_device: &DeviceType, cl_device: &DeviceType, mem: &mut SharedTensor<u8>) {
    sync_back_and_forth(b, 256, nt_device, cl_device, mem)
}

#[bench]
#[cfg(feature = "cuda")]
fn bench_2_sync_128mb_native_cuda(b: &mut Bencher) {
    let cuda_backend = cuda_backend();
    let cl_device = cuda_backend.device();
    let native_backend = native_backend();
    let nt_device = native_backend.device();
    // if let &DeviceType::Cuda(ref cl_d) = cl_device {
    //     println!("{:?}", cl_d.hardwares()[0].clone().load_name());
    // }
    let mem = &mut SharedTensor::<u8>::new(&(128 * 1_048_576));
    mem.write_only(&cl_device);
    bench_2_sync_128mb_native_cuda_profile(b, nt_device, cl_device, mem);
}

#[inline(never)]
#[cfg(feature = "cuda")]
fn bench_2_sync_128mb_native_cuda_profile(b: &mut Bencher, nt_device: &DeviceType, cl_device: &DeviceType, mem: &mut SharedTensor<u8>) {
    sync_back_and_forth(b, 2, nt_device, cl_device, mem)
}
