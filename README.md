# rust_partdiff
Program for calculation of partial differential equations implemented in Rust

# building the crate
To build the crate correctly currently some features have to be activated.
## Features
* Decide which indexing implementation is used for PartdiffMatrix
  * feature: `2d-array-indexing` has a syntax of `matrix[[x,y]]` and should provide better performance
  * feature: `C-style-indexing` has a syntax of `matrix[x][y]` and has probably worse performance due to the internals of the implementation 
* Decide whether bounds checking should be used for matrix access
  * feature: `unsafe-indexing` uses the unsafe `get_unchecked` methods of Vec and does no bounds checking but therefore performs better
  * not building with `unsafe-indexing` defaults to using access methods of Vec that apply bounds checking

The current default way to build the crate for performance should be: `cargo build --release --features "2d-array-indexing,unsafe-indexing"` 
