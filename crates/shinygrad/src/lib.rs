//! # shinygrad
//!
//! A modular deep learning framework in Rust, inspired by [Burn](https://github.com/tracel-ai/burn).
//! Built from scratch as a learning project with first-class support for multiple backends.
//!
//! ## Architecture
//!
//! Shinygrad follows Burn's crate structure to provide a clean, extensible design:
//!
//! | Crate | Burn equivalent | Role |
//! |---|---|---|
//! | `shinygrad-backend` | `burn-backend` | `Backend` trait, `FloatOps`, `DType`, `Shape`, `Device` |
//! | `shinygrad-tensor` | `burn-tensor` | `Tensor<B, D>` — user-facing tensor API |
//! | `shinygrad-autodiff` | `burn-autodiff` | `Autodiff<B>` decorator, computation graph, backward pass |
//! | `shinygrad-core` | `burn-core` | `Module` trait, `Param` |
//! | `shinygrad-nn` | `burn-nn` | `Linear`, `ReLU`, `Sequential`, loss functions, initializers |
//! | `shinygrad-optim` | `burn-optim` | `SGD`, `Adam`, `Momentum` |
//! | `shinygrad-cpu-scratch` | `burn-ndarray` | Handwritten CPU backend via `ndarray` |
//! | `shinygrad-cubecl` | `burn-cubecl` | Generic CubeCL backend (GPU/CPU JIT) |
//! | `shinygrad-cpu` | `burn-cpu` | CubeCL CPU runtime wrapper |
//! | `shinygrad-cuda` | `burn-cuda` | CubeCL CUDA runtime wrapper |
//! | `shinygrad` (this crate) | `burn` | Facade — re-exports everything |
//!
//! ## Backend abstraction
//!
//! | Backend | Feature flag | Status |
//! |---|---|---|
//! | `CpuBackend<E>` (scratch) | `cpu-scratch` (default) | Ready |
//! | `CubeClCpuBackend<E>` | `cubecl-cpu` | Ready |
//! | `CubeClCudaBackend<E>` | `cubecl-cuda` | Planned |
//!
//! ## Running examples
//!
//! ```bash
//! cargo run --example xor --release
//! cargo run --example mnist_mini --release
//! ```
//!
//! ## Testing
//!
//! ```bash
//! cargo test                           # cpu-scratch backend (default)
//! cargo test -F cubecl-cpu             # CubeCL CPU
//! ```

pub mod prelude {
    pub use shinygrad_backend::{
        Backend, AutodiffBackend, BackendTypes,
        FloatElement, FloatOps,
        Device, DType, Shape, TensorData,
        tensor::FloatTensor,
    };
    pub use shinygrad_tensor::Tensor;
    pub use shinygrad_autodiff::Autodiff;
    pub use shinygrad_core::module::{Module, Param};
    pub use shinygrad_nn::{
        Linear, LinearConfig,
        Sequential,
        ReLU, Softmax,
        kaiming_uniform, xavier_uniform,
        CrossEntropyLoss,
    };
    pub use shinygrad_optim::{SGD, Adam, Momentum, Optimizer, SgdState, AdamState, MomentumState};

    #[cfg(feature = "cpu-scratch")]
    pub use shinygrad_cpu_scratch::CpuBackend;

    #[cfg(feature = "cubecl")]
    pub use shinygrad_cubecl::CubeBackend;
    #[cfg(feature = "cubecl-cpu")]
    pub use shinygrad_cpu::Cpu as CubeClCpuBackend;
    #[cfg(feature = "cubecl-cuda")]
    pub use shinygrad_cuda::Cuda as CubeClCudaBackend;
}
