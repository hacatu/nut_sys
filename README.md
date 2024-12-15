# `nut_sys`

`nut_sys` is a low-level Rust wrapper for the [Number Theory Utils](https://github.com/hacatu/Number-Theory-Utils) (`nut`) C library.  It provides raw FFI bindings and some safe wrappers.  Not all functions from the C library are currently exported, more will be added as needed.

## Build Options
- **Use Preinstalled Library**: Link to an already installed version of the `nut` library.
- **Automatically Build from Source**: Download and compile the `nut` C library from a submodule (requires `--features build-c`).

## Installation

### Using an Installed `nut` Library
If you already have the `nut` library installed on your system, you might need to set the `NUT_LIB_DIR` environment variable if it is in a non-standard location.
Then just run `cargo build` as normal:

```bash
export NUT_LIB_DIR=/path/to/nut/lib
cargo build
```

For example, the default install location is `/usr/local/lib/`, which is usually in the library path, but not for some linkers like `mold`.

### Automatically Building `nut` from Source
If you do not have `nut` installed and do not want to manually build it:

1. Clone the repository and initialize the submodule:
   ```bash
   git submodule update --recursive --init
   ```

2. Build the Rust crate with the `build-c` feature enabled:
   ```bash
   cargo build --features build-c
   ```

This will compile the `nut` library and link it statically to your Rust crate.

You can also then install or run more tests on the copy of `nut` that cargo downloaded, but be careful because
changing the `<repository>/nut/` directory can cause `build.rs` to re-run.

## Usage
Add `nut_sys` to your `Cargo.toml`:

```toml
[dependencies]
nut_sys = "0.1.0" # Replace with the latest version from crates.io
```

Then, in your Rust code:

```rust
use nut_sys::{sieve_largest_factors, Factors};

fn main() -> Result<(), ThinBox<dyn Error>> {
	let largest_factors = sieve_largest_factors(100_0000u64);
	let largest_factors: &[_] = largest_factors.borrow();
	let mut count = 0;
	let mut fxn = Factors::make_ub(100_000);
	for n in 1..=100_000 {
		fxn.fill_from_largest_factors(n, largest_factors);
		if (0..fxn.num_primes()).all(|i|fxn[i].power == 1) {
			count += 1;
		}
	}
	println!("Found {count} squarefree numbers up to 100_000");
	Ok(())
}
```

## Links
- **Crate**: [https://crates.io/crates/nut_sys](https://crates.io/crates/nut_sys)  
- **C Library Repository**: [https://github.com/hacatu/Number-Theory-Utils](https://github.com/hacatu/Number-Theory-Utils)  
- **Documentation**: [https://docs.rs/nut_sys](https://docs.rs/nut_sys)  

## License
`nut_sys` and `nut` are both licensed under [MPL-2.0](https://opensource.org/licenses/mpl-2-0). See `LICENSE` for details.

