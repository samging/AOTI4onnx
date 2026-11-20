use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    let libtorch = PathBuf::from(
        env::var("LIBTORCH").expect("set LIBTORCH to the libtorch root (contains lib/)"),
    );
    let lib = libtorch.join("lib");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());

    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-lib=dylib=c++"),
        Ok("linux") => println!("cargo:rustc-link-lib=dylib=stdc++"),
        _ => {}
    }
}
