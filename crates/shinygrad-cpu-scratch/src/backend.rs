use std::marker::PhantomData;
use std::sync::Arc;

use ndarray::{ArrayD, Axis, IxDyn};
use num_traits::{Zero, One};
use shinygrad_backend::{
    Backend, BackendTypes,
    Shape, Device, FloatElement,
    FloatOps,
    TensorData,
    tensor::FloatTensor,
};

#[derive(Debug, Clone)]
pub struct CpuTensorPrimitive<E: FloatElement> {
    pub data: Arc<ArrayD<E>>,
}

impl<E: FloatElement> CpuTensorPrimitive<E> {
    pub fn new(data: ArrayD<E>) -> Self {
        CpuTensorPrimitive {
            data: Arc::new(data),
        }
    }

    pub fn shape(&self) -> Shape {
        Shape::new(self.data.shape().to_vec())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuBackend<E: FloatElement> {
    _elem: PhantomData<E>,
}

impl<E: FloatElement> Default for CpuBackend<E> {
    fn default() -> Self {
        CpuBackend { _elem: PhantomData }
    }
}

impl<E: FloatElement> CpuBackend<E> {
    pub fn new() -> Self {
        CpuBackend { _elem: PhantomData }
    }
}

impl<E: FloatElement> BackendTypes for CpuBackend<E> {
    type Device = Device;
    type FloatTensorPrimitive = CpuTensorPrimitive<E>;
    type FloatElem = E;
}

impl<E> Backend for CpuBackend<E>
where
    E: FloatElement + ndarray::ScalarOperand + Zero + One,
{
    fn name(_device: &Self::Device) -> String {
        format!("cpu-{:?}", E::dtype())
    }

    fn seed(_device: &Self::Device, _seed: u64) {}
}

impl<E> FloatOps<CpuBackend<E>> for CpuBackend<E>
where
    E: FloatElement + ndarray::ScalarOperand + Zero + One,
{
    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new((*lhs.data).clone() + (*rhs.data).clone())
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new((*lhs.data).clone() * (*rhs.data).clone())
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new((*lhs.data).clone() - (*rhs.data).clone())
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new((*lhs.data).clone() / (*rhs.data).clone())
    }

    fn float_pow(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let result = lhs.data.iter().zip(rhs.data.iter())
            .map(|(&a, &b)| a.powf(b))
            .collect::<Vec<E>>();
        let shape = lhs.data.shape().to_vec();
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), result).unwrap();
        CpuTensorPrimitive::new(arr)
    }

    fn float_pow_scalar(lhs: FloatTensor<Self>, exponent: f64) -> FloatTensor<Self> {
        let e = E::from_f64(exponent);
        CpuTensorPrimitive::new(lhs.data.mapv(|x| x.powf(e)))
    }

    fn float_neg(value: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new(value.data.mapv(|x| -x))
    }

    fn float_exp(value: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new(value.data.mapv(|x| x.exp()))
    }

    fn float_log(value: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new(value.data.mapv(|x| x.ln()))
    }

    fn float_relu(value: FloatTensor<Self>) -> FloatTensor<Self> {
        CpuTensorPrimitive::new(value.data.mapv(|x| {
            let z = <E as FloatElement>::zero();
            if x < z { z } else { x }
        }))
    }

    fn float_sum(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let sum = value.data.sum();
        let arr = ArrayD::from_shape_vec(IxDyn(&[1]), vec![sum]).unwrap();
        CpuTensorPrimitive::new(arr)
    }

    fn float_sum_dim(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let summed = value.data.sum_axis(Axis(dim));
        CpuTensorPrimitive::new(summed.insert_axis(Axis(dim)))
    }

    fn float_mean_dim(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let summed = value.data.sum_axis(Axis(dim));
        let n = value.data.shape()[dim] as f64;
        let result = summed.mapv(|x| E::from_f64(x.to_f64() / n));
        CpuTensorPrimitive::new(result.insert_axis(Axis(dim)))
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let l = (*lhs.data).clone();
        let r = (*rhs.data).clone();
        let l2d: ndarray::Array2<E> = l.into_dimensionality::<ndarray::Ix2>().unwrap();
        let r2d: ndarray::Array2<E> = r.into_dimensionality::<ndarray::Ix2>().unwrap();
        CpuTensorPrimitive::new(l2d.dot(&r2d).into_dyn())
    }

    fn float_transpose(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let v = (*value.data).clone();
        if v.ndim() == 2 {
            let v2d = v.into_dimensionality::<ndarray::Ix2>().unwrap();
            CpuTensorPrimitive::new(v2d.t().to_owned().into_dyn())
        } else {
            let mut axes: Vec<usize> = (0..v.ndim()).collect();
            axes.reverse();
            CpuTensorPrimitive::new(v.permuted_axes(axes).as_standard_layout().to_owned())
        }
    }

    fn float_softmax(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let data = (*value.data).clone();
        let max_vals = data.map_axis(Axis(dim), |row| {
            row.iter().cloned().fold(E::min_value(), |a, b| if a > b { a } else { b })
        });
        let shifted = data - &max_vals.insert_axis(Axis(dim));
        let exp_vals = shifted.mapv(|x| x.exp());
        let sum_vals = exp_vals.sum_axis(Axis(dim));
        CpuTensorPrimitive::new(exp_vals / &sum_vals.insert_axis(Axis(dim)))
    }

    fn float_reshape(value: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let arr = (*value.data).clone().into_shape(IxDyn(&shape.dims)).unwrap();
        CpuTensorPrimitive::new(arr)
    }

    fn float_broadcast_to(value: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let data = (*value.data).clone();
        let result = data.broadcast(IxDyn(&shape.dims)).unwrap().to_owned();
        CpuTensorPrimitive::new(result)
    }

    fn float_shape(value: &FloatTensor<Self>) -> Shape {
        value.shape()
    }

    fn float_into_data(value: FloatTensor<Self>) -> TensorData<E> {
        let shape = value.shape();
        let data = value.data.as_slice().unwrap().to_vec();
        TensorData::new(data, shape)
    }

    fn float_zeros(_device: &Device, shape: Shape) -> FloatTensor<Self> {
        CpuTensorPrimitive::new(ArrayD::zeros(IxDyn(&shape.dims)))
    }

    fn float_ones(_device: &Device, shape: Shape) -> FloatTensor<Self> {
        CpuTensorPrimitive::new(ArrayD::ones(IxDyn(&shape.dims)))
    }

    fn float_from_data(_device: &Device, data: TensorData<E>) -> FloatTensor<Self> {
        let arr = ArrayD::from_shape_vec(IxDyn(&data.shape.dims), data.value).unwrap();
        CpuTensorPrimitive::new(arr)
    }
}
