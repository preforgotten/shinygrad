//! # shinygrad-cuda
//!
//! Thin wrapper around `shinygrad-cubecl` using `cubecl::cuda::CudaRuntime`.

pub use cubecl::cuda::CudaDevice;

/// CubeCL CUDA backend type alias.
pub type Cuda<E> = shinygrad_cubecl::CubeBackend<cubecl::cuda::CudaRuntime, E>;
