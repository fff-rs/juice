extern crate coaster;
use coaster::prelude::*;

fn main() {
    let ntv = Native::new();
    let dev = ntv.new_device(ntv.hardwares()).unwrap();

    let x = &mut SharedTensor::<f32>::new(&10);
    let m1 = x.write_only(&dev).unwrap();
    let m2 = x.write_only(&dev).unwrap();
    //~^ ERROR error: cannot borrow `*x` as mutable more than once at a time
}

