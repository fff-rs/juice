#![cfg(feature = "nightly")]
#![feature(test)]

use co::backend::{Backend, BackendConfig};
use co::device::IDevice;
use co::framework::IFramework;
use co::tensor::SharedTensor;
use coaster as co;
use test::Bencher;

#[cfg(feature = "cuda")]
use co::frameworks::Cuda;
#[cfg(feature = "native")]
use co::frameworks::Native;
#[cfg(feature = "opencl")]
use co::frameworks::OpenCL;

#[cfg(feature = "native")]
fn native_backend() -> Backend<Native> {
    let framework = Native::new();
    let hardwares = framework.hardwares().to_vec();
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

#[cfg(feature = "cuda")]
use co::frameworks::cuda::get_cuda_backend as cuda_backend;

fn sync_back_and_forth<F1, F2>(
    b: &mut Bencher,
    backend1: Backend<F1>,
    backend2: Backend<F2>,
    mem_size: usize,
) where
    F1: co::IFramework + Clone,
    F2: co::IFramework + Clone,
{
    let dev1 = backend1.device();
    let dev2 = backend2.device();

    let mem = &mut SharedTensor::<u8>::new(&mem_size);
    /* initialize and warm-up */
    mem.write_only(dev2).unwrap();
    mem.read_write(dev1).unwrap();
    mem.read_write(dev2).unwrap();

    b.bytes = mem_size as u64 * 2; // we do two transfers per iteration
    b.iter(|| {
        mem.read_write(dev1).unwrap();
        mem.read_write(dev2).unwrap();
    });
}

fn unidirectional_sync<F1, F2>(
    b: &mut Bencher,
    src_backend: Backend<F1>,
    dst_backend: Backend<F2>,
    mem_size: usize,
) where
    F1: co::IFramework + Clone,
    F2: co::IFramework + Clone,
{
    let src_dev = src_backend.device();
    let dst_dev = dst_backend.device();

    let mem = &mut SharedTensor::<u8>::new(&mem_size);
    /* initialize and warm-up */
    mem.write_only(src_dev).unwrap();
    mem.read(dst_dev).unwrap();

    b.bytes = mem_size as u64;
    b.iter(|| {
        mem.write_only(src_dev).unwrap();
        mem.read(dst_dev).unwrap();
    });
}

#[cfg(feature = "native")]
#[cfg(feature = "opencl")]
mod opencl_and_native {
    use super::{native_backend, opencl_backend, sync_back_and_forth, unidirectional_sync};
    use co::device::IDevice;
    use co::frameworks::opencl;
    use test::Bencher;

    #[inline(never)]
    fn bench_256_alloc_1mb_opencl_profile(b: &mut Bencher, device: &opencl::Context, size: usize) {
        b.iter(|| {
            for _ in 0..256 {
                device.alloc_memory(size).unwrap();
            }
        });
    }

    #[bench]
    fn bench_256_alloc_1mb_opencl(b: &mut Bencher) {
        let opencl_backend = opencl_backend();
        let cl_device = opencl_backend.device();
        bench_256_alloc_1mb_opencl_profile(b, cl_device, 1_048_576);
    }

    #[bench]
    fn bench_sync_1kb_native_opencl_back_and_forth(b: &mut Bencher) {
        sync_back_and_forth(b, opencl_backend(), native_backend(), 1024);
    }

    #[bench]
    fn bench_sync_1kb_native_to_opencl(b: &mut Bencher) {
        unidirectional_sync(b, native_backend(), opencl_backend(), 1024);
    }

    #[bench]
    fn bench_sync_1kb_opencl_to_native(b: &mut Bencher) {
        unidirectional_sync(b, opencl_backend(), native_backend(), 1024);
    }

    #[bench]
    fn bench_sync_1mb_native_opencl_back_and_forth(b: &mut Bencher) {
        sync_back_and_forth(b, opencl_backend(), native_backend(), 1_048_576);
    }

    #[bench]
    fn bench_sync_1mb_native_to_opencl(b: &mut Bencher) {
        unidirectional_sync(b, native_backend(), opencl_backend(), 1_048_576);
    }

    #[bench]
    fn bench_sync_1mb_opencl_to_native(b: &mut Bencher) {
        unidirectional_sync(b, opencl_backend(), native_backend(), 1_048_576);
    }

    #[bench]
    fn bench_sync_128mb_native_opencl_back_and_forth(b: &mut Bencher) {
        sync_back_and_forth(b, opencl_backend(), native_backend(), 128 * 1_048_576);
    }

    #[bench]
    fn bench_sync_128mb_native_to_opencl(b: &mut Bencher) {
        unidirectional_sync(b, native_backend(), opencl_backend(), 128 * 1_048_576);
    }

    #[bench]
    fn bench_sync_128mb_opencl_to_native(b: &mut Bencher) {
        unidirectional_sync(b, opencl_backend(), native_backend(), 128 * 1_048_576);
    }
}

#[cfg(feature = "native")]
#[cfg(feature = "cuda")]
mod cuda_and_native {
    use super::{cuda_backend, native_backend, sync_back_and_forth, unidirectional_sync};
    use test::Bencher;

    #[bench]
    fn bench_sync_1kb_native_cuda_back_and_forth(b: &mut Bencher) {
        sync_back_and_forth(b, cuda_backend(), native_backend(), 1024);
    }

    #[bench]
    fn bench_sync_1kb_native_to_cuda(b: &mut Bencher) {
        unidirectional_sync(b, native_backend(), cuda_backend(), 1024);
    }

    #[bench]
    fn bench_sync_1kb_cuda_to_native(b: &mut Bencher) {
        unidirectional_sync(b, cuda_backend(), native_backend(), 1024);
    }

    #[bench]
    fn bench_sync_1mb_native_cuda_back_and_forth(b: &mut Bencher) {
        sync_back_and_forth(b, cuda_backend(), native_backend(), 1_048_576);
    }

    #[bench]
    fn bench_sync_1mb_native_to_cuda(b: &mut Bencher) {
        unidirectional_sync(b, native_backend(), cuda_backend(), 1_048_576);
    }

    #[bench]
    fn bench_sync_1mb_cuda_to_native(b: &mut Bencher) {
        unidirectional_sync(b, cuda_backend(), native_backend(), 1_048_576);
    }

    #[bench]
    fn bench_sync_128mb_native_cuda_back_and_forth(b: &mut Bencher) {
        sync_back_and_forth(b, cuda_backend(), native_backend(), 128 * 1_048_576);
    }

    #[bench]
    fn bench_sync_128mb_native_to_cuda(b: &mut Bencher) {
        unidirectional_sync(b, native_backend(), cuda_backend(), 128 * 1_048_576);
    }

    #[bench]
    fn bench_sync_128mb_cuda_to_native(b: &mut Bencher) {
        unidirectional_sync(b, cuda_backend(), native_backend(), 128 * 1_048_576);
    }
}

#[bench]
#[cfg(feature = "cuda")]
fn bench_shared_tensor_access_time_first(b: &mut Bencher) {
    let cuda_backend = cuda_backend();
    let cu_device = cuda_backend.device();
    let native_backend = native_backend();
    let nt_device = native_backend.device();

    let mut x = SharedTensor::<f32>::new(&[128]);
    x.write_only(nt_device).unwrap();
    x.write_only(cu_device).unwrap();
    x.read(nt_device).unwrap();

    b.iter(|| x.read(nt_device).unwrap())
}

#[bench]
#[cfg(feature = "cuda")]
fn bench_shared_tensor_access_time_second(b: &mut Bencher) {
    let cuda_backend = cuda_backend();
    let cu_device = cuda_backend.device();
    let native_backend = native_backend();
    let nt_device = native_backend.device();

    let mut x = SharedTensor::<f32>::new(&[128]);
    x.write_only(cu_device).unwrap();
    x.write_only(nt_device).unwrap();

    b.iter(|| x.read(nt_device).unwrap())
}
