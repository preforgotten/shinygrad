//! # shinygrad-cpu
//!
//! Thin wrapper around `shinygrad-cubecl` using `cubecl::cpu::CpuRuntime`.

pub use cubecl::cpu::CpuDevice;

/// CubeCL CPU backend type alias.
pub type Cpu<E> = shinygrad_cubecl::CubeBackend<cubecl::cpu::CpuRuntime, E>;
