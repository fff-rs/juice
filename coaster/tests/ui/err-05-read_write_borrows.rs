use coaster::prelude::*;

fn main() {
    let ntv = Native::new();
    let dev = ntv.new_device(ntv.hardwares()).unwrap();

    let x = &mut SharedTensor::<f32>::new(&10);
    let m1 = x.write_only(&dev).unwrap();
    let m2 = x.read(&dev).unwrap();
    //~^ ERROR cannot borrow `*x` as immutable because it is also borrowed as mutable

    // need additional bindings, so rust knows it's used afterwards
    let _foo = m1;
    let _bar = m2;
}
