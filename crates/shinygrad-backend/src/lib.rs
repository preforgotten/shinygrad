//! # shinygrad-backend
//!
//! Core abstractions for tensor operations across different hardware backends.
//! This crate defines the traits that every backend must implement — it has
//! zero runtime dependencies.
//!
//! ## Key traits
//!
//! - **`BackendTypes`** — associated types: `Device`, `FloatTensorPrimitive`, `FloatElem`
//! - **`FloatOps<B>`** — low-level float tensor operations (`add`, `mul`, `matmul`, `relu`, ...)
//! - **`Backend`** — combines `BackendTypes` + `FloatOps<Self>` + lifecycle methods
//! - **`AutodiffBackend`** — extends `Backend` with gradient tracking and backward pass
//!
//! ## Key types
//!
//! - **`FloatTensor<B>`** — type alias for `B::FloatTensorPrimitive`
//! - **`Device`** — `Cpu` (future: `Gpu`, `Accelerator`)
//! - **`DType`** — `F32`, `F64`
//! - **`Shape`** — dynamic shape `Vec<usize>`
//! - **`TensorData<E>`** — host-accessible tensor data used for device↔host transfers
//! - **`FloatElement`** — trait for element types (`f32`, `f64`) with math operations
//!
//! ## Example: implementing a custom backend
//!
//! ```ignore
//! use shinygrad_backend::*;
//!
//! impl BackendTypes for MyBackend {
//!     type Device = Device;
//!     type FloatTensorPrimitive = MyTensor;
//!     type FloatElem = f32;
//! }
//!
//! impl FloatOps<MyBackend> for MyBackend {
//!     fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
//!         // delegate to your underlying library
//!     }
//!     // ... implement all required methods
//! }
//!
//! impl Backend for MyBackend {
//!     fn name(device: &Self::Device) -> String { "my-backend".into() }
//!     fn seed(device: &Self::Device, seed: u64) { /* ... */ }
//! }
//! ```

pub mod dtype;
pub mod element;
pub mod shape;
pub mod device;
pub mod data;
pub mod backend;
pub mod tensor;

pub use dtype::DType;
pub use element::FloatElement;
pub use shape::Shape;
pub use device::Device;
pub use data::TensorData;
pub use backend::{
    Backend, BackendTypes, AutodiffBackend,
    ExecutionError,
    ops::float::FloatOps,
};
pub use tensor::FloatTensor;
