//! Multi-backend test harness for shinygrad.
//!
//! Each backend gets its own `#[cfg(feature = "...")]` module generated
//! via the `backend_tests!` macro.
//!
//! Run: `cargo test` (cpu-scratch), `cargo test -F cubecl-cpu` (CubeCL CPU)

use shinygrad::prelude::*;

fn to_elem_vec<B: Backend>(values: &[f64]) -> Vec<B::FloatElem>
where B::FloatElem: FloatElement
{
    values.iter().map(|&v| B::FloatElem::from_f64(v)).collect()
}

fn tensor1d<B: Backend>(device: &<B as BackendTypes>::Device, data: &[f64]) -> Tensor<B, 1>
where B::FloatElem: FloatElement
{
    let elems = to_elem_vec::<B>(data);
    Tensor::<B, 1>::from_data(device, TensorData::new(elems, Shape::new(vec![data.len()])))
}

fn tensor2d<B: Backend>(device: &<B as BackendTypes>::Device, data: &[f64], rows: usize, cols: usize) -> Tensor<B, 2>
where B::FloatElem: FloatElement
{
    let elems = to_elem_vec::<B>(data);
    Tensor::<B, 2>::from_data(device, TensorData::new(elems, Shape::new(vec![rows, cols])))
}

fn assert_tensor_eq<B: Backend>(tensor: &Tensor<B, 1>, expected: &[f64])
where B::FloatElem: FloatElement
{
    let data = tensor.clone().into_data();
    assert_eq!(data.value.len(), expected.len(), "tensor length mismatch");
    for (i, (&got, &exp)) in data.value.iter().zip(expected.iter()).enumerate() {
        let got_f64 = got.to_f64();
        assert!((got_f64 - exp).abs() < 1e-6, "index {i}: expected {exp}, got {got_f64}");
    }
}

fn test_add<B: Backend>(device: &<B as BackendTypes>::Device) where B::FloatElem: FloatElement {
    let a = tensor1d::<B>(device, &[1.0]);
    let b = tensor1d::<B>(device, &[2.0]);
    assert_tensor_eq::<B>(&a.add(b), &[3.0]);
}

fn test_sub<B: Backend>(device: &<B as BackendTypes>::Device) where B::FloatElem: FloatElement {
    let a = tensor1d::<B>(device, &[5.0]);
    let b = tensor1d::<B>(device, &[3.0]);
    assert_tensor_eq::<B>(&a.sub(b), &[2.0]);
}

fn test_mul<B: Backend>(device: &<B as BackendTypes>::Device) where B::FloatElem: FloatElement {
    let a = tensor1d::<B>(device, &[3.0]);
    let b = tensor1d::<B>(device, &[4.0]);
    assert_tensor_eq::<B>(&a.mul(b), &[12.0]);
}

fn test_sum<B: Backend>(device: &<B as BackendTypes>::Device) where B::FloatElem: FloatElement {
    let a = tensor1d::<B>(device, &[1.0, 2.0, 3.0]);
    let data = a.sum().into_data();
    assert_eq!(data.value.len(), 1);
    assert!((data.value[0].to_f64() - 6.0).abs() < 1e-6);
}

fn test_shape<B: Backend>(device: &<B as BackendTypes>::Device) where B::FloatElem: FloatElement {
    let t = tensor2d::<B>(device, &[1.0, 2.0, 3.0, 4.0], 2, 2);
    let data = t.into_data();
    assert_eq!(data.shape.dims.len(), 2);
    assert_eq!(data.shape.dims[0], 2);
    assert_eq!(data.shape.dims[1], 2);
    for (i, (got, exp)) in data.value.iter().zip(&[1.0, 2.0, 3.0, 4.0]).enumerate() {
        assert!((got.to_f64() - exp).abs() < 1e-6, "index {i}: expected {exp}, got {}", got.to_f64());
    }
}

fn test_zeros<B: Backend>(device: &<B as BackendTypes>::Device) where B::FloatElem: FloatElement {
    let t = Tensor::<B, 1>::zeros(device, Shape::new(vec![3]));
    for v in &t.into_data().value {
        assert!((v.to_f64() - 0.0).abs() < 1e-6);
    }
}

macro_rules! backend_tests {
    ($backend:ty, $modname:ident, $dev:expr) => {
        mod $modname {
            use super::*;
            fn device() -> <$backend as BackendTypes>::Device { $dev }
            #[test] fn add() { test_add::<$backend>(&device()); }
            #[test] fn sub() { test_sub::<$backend>(&device()); }
            #[test] fn mul() { test_mul::<$backend>(&device()); }
            #[test] fn sum() { test_sum::<$backend>(&device()); }
            #[test] fn shape() { test_shape::<$backend>(&device()); }
            #[test] fn zeros() { test_zeros::<$backend>(&device()); }
        }
    };
}

#[cfg(feature = "cpu-scratch")]
backend_tests!(CpuBackend<f32>, cpu_scratch_f32, Device::Cpu);
#[cfg(feature = "cpu-scratch")]
backend_tests!(CpuBackend<f64>, cpu_scratch_f64, Device::Cpu);

#[cfg(feature = "cubecl-cpu")]
backend_tests!(CubeClCpuBackend<f32>, cubecl_cpu_f32, shinygrad_cpu::CpuDevice::default());
#[cfg(feature = "cubecl-cpu")]
backend_tests!(CubeClCpuBackend<f64>, cubecl_cpu_f64, shinygrad_cpu::CpuDevice::default());

// #[cfg(feature = "cubecl-cuda")]
// backend_tests!(CubeClCudaBackend<f32>, cubecl_cuda_f32, shinygrad_cuda::CudaDevice::default());
