//! # shinygrad-autodiff
//!
//! Automatic differentiation via a backend decorator.
//!
//! `Autodiff<B>` wraps any `Backend` and transparently adds gradient tracking.
//! Every tensor operation through `Autodiff<B>` builds a computation graph
//! (dynamic, eager-mode), and calling `.backward()` computes gradients via
//! reverse-mode topological sort — the same algorithm as micrograd/PyTorch.
//!
//! ## Architecture
//!
//! ```text
//! User code                 Backend chain
//! ─────────                 ─────────────
//! Tensor<Autodiff<B>, D>
//!   │                       Autodiff<B>  ← decorator (tracks graph)
//!   │                         │
//!   │ delegates              B           ← inner backend (actual compute)
//!   ▼
//! Backend::float_add(autodiff_tensor, autodiff_tensor)
//!   → 1. B::float_add(inner, inner)      // compute result
//!   → 2. graph.add_node(op, inputs)       // track backward
//!   → 3. return ADTensor { data, node }
//! ```
//!
//! ## Key types
//!
//! - **`Autodiff<B>`** — the backend decorator implementing both `Backend` and `AutodiffBackend`
//! - **`AutodiffTensor<B>`** — wraps `B::FloatTensorPrimitive` + a computation graph node
//! - **`Gradients<B>`** — map from node ID → gradient tensor after backward
//!
//! ## Usage
//!
//! ```ignore
//! use shinygrad_autodiff::Autodiff;
//! use shinygrad_backend::AutodiffBackend;
//!
//! type MyBackend = Autodiff<CpuBackend<f64>>;
//!
//! let loss = compute_loss(...);
//! let grads: Gradients<CpuBackend<f64>> = MyBackend::backward(loss.into_primitive());
//! ```

mod backend;
mod tensor;
mod graph;

pub use backend::Autodiff;
pub use tensor::AutodiffTensor;
pub use graph::Gradients;
