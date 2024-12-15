use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let build_c = env::var("CARGO_FEATURE_BUILD_C").is_ok();
    if build_c {
        // Path to the C library submodule
        let nut_dir = PathBuf::from("nut");
        if !nut_dir.exists() {
            panic!("The 'nut' submodule is missing. Did you forget to initialize it?  Either do `git submodule update --recursive --init`, or manually build and install nut.");
        }
        let status = Command::new("./waf")
            .arg("configure")
            .current_dir(&nut_dir)
            .status()
            .expect("Failed to run `./waf configure`");
        if !status.success() {
            panic!("`./waf configure` failed");
        }
        let status = Command::new("./waf")
            .arg("build_release")
            .current_dir(&nut_dir)
            .status()
            .expect("Failed to run `./waf build_release`");
        if !status.success() {
            panic!("`./waf build_release` failed");
        }
        // Tell Cargo where to find the built library, which will be in `<nut>/build/<variant>/libnut.a`
        let lib_dir = nut_dir.join("build").join("release");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=static=nut");
    } else {
        // Use an existing installed copy of the C library
        println!("cargo:rustc-link-lib=nut");
    }
    println!("cargo:rerun-if-changed=nut/");
}

