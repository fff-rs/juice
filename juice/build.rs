fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("capnp")
        .file("capnp/juice.capnp")
        .run()
        .expect("capnpc schema compiler command must succeed");
}
