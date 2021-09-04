use coaster::prelude::*;

fn main() {
    let ntv = Native::new();
    let dev = ntv.new_device(ntv.hardwares()).unwrap();

    let x = &mut SharedTensor::<f32>::new(&10);
    x.write_only(&dev).unwrap();

    let _m1 = x.read(&dev);
    let _m2 = x.read(&dev);
    let _m3 = x.read(&dev);
}
