//! # shinygrad-tensor
//!
//! High-level tensor API generic over any `Backend`.
//!
//! `Tensor<B, D>` wraps a backend's `FloatTensorPrimitive` and exposes
//! a familiar eager-mode API: `add`, `mul`, `matmul`, `relu`, `softmax`, etc.
//! Every operation is dispatched through the backend's `FloatOps` implementation.
//!
//! ## Const generics
//!
//! The number of dimensions is tracked at compile time via `const D: usize`:
//!
//! ```ignore
//! use shinygrad_tensor::Tensor;
//!
//! let x: Tensor<MyBackend, 2> = Tensor::zeros(&device, Shape::new(vec![32, 128]));
//! let w: Tensor<MyBackend, 2> = Tensor::zeros(&device, Shape::new(vec![128, 64]));
//! let y = x.matmul(w);  // still Tensor<MyBackend, 2>
//! ```
//!
//! ## Data transfer
//!
//! ```ignore
//! // Host → device
//! let t = Tensor::from_data(&device, TensorData::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2])));
//!
//! // Device → host
//! let data: TensorData<f32> = t.into_data();
//! ```

mod tensor;

pub use tensor::Tensor;
