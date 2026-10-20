//! # shinygrad-cubecl
//!
//! CubeCL backend — implements `Backend` + `FloatOps` using CubeCL JIT kernels.
//!
//! ## Architecture
//!
//! ```ignore
//! CubeBackend<R: Runtime>      ← generic over CubeCL runtime (CpuRuntime, CudaRuntime, ...)
//!   └── CubeTensorPrimitive<R> ← wraps cubecl::server::Handle + shape metadata
//! ```
//!
//! Each tensor operation is a `#[cube]` kernel compiled JIT to the target platform.
//!
//! ## Status
//!
//! Core ops implemented: zeros, from_data, into_data, add, sub, mul, sum, shape.
//! Remaining ops (matmul, relu, softmax, transpose, etc.) to be added incrementally.

mod backend;
mod tensor;

pub use backend::CubeBackend;
pub use tensor::CubeTensorPrimitive;
