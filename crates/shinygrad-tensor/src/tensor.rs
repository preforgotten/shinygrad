use std::marker::PhantomData;

use shinygrad_backend::{
    Backend,
    Shape,
    TensorData,
    tensor::FloatTensor,
};

#[derive(Debug, Clone)]
pub struct Tensor<B: Backend, const D: usize> {
    pub primitive: FloatTensor<B>,
    _dim: PhantomData<[usize; D]>,
}

impl<B: Backend, const D: usize> Tensor<B, D> {
    pub fn new(primitive: FloatTensor<B>) -> Self {
        Tensor {
            primitive,
            _dim: PhantomData,
        }
    }

    pub fn from_primitive(primitive: FloatTensor<B>) -> Self {
        Tensor {
            primitive,
            _dim: PhantomData,
        }
    }

    pub fn into_primitive(self) -> FloatTensor<B> {
        self.primitive
    }

    pub fn shape(&self) -> Shape {
        B::float_shape(&self.primitive)
    }

    pub fn dims(&self) -> [usize; D] {
        let shape = self.shape();
        let mut dims = [0; D];
        for (i, d) in shape.dims.iter().enumerate() {
            dims[i] = *d;
        }
        dims
    }

    pub fn into_data(self) -> TensorData<B::FloatElem> {
        B::float_into_data(self.primitive)
    }

    pub fn from_data(device: &<B as shinygrad_backend::BackendTypes>::Device, data: TensorData<B::FloatElem>) -> Self {
        Self::from_primitive(B::float_from_data(device, data))
    }

    pub fn zeros(device: &<B as shinygrad_backend::BackendTypes>::Device, shape: Shape) -> Self {
        Self::from_primitive(B::float_zeros(device, shape))
    }

    pub fn ones(device: &<B as shinygrad_backend::BackendTypes>::Device, shape: Shape) -> Self {
        Self::from_primitive(B::float_ones(device, shape))
    }

    pub fn scalar(device: &<B as shinygrad_backend::BackendTypes>::Device, value: B::FloatElem) -> Self {
        Self::from_primitive(B::float_scalar(device, value))
    }

    pub fn add(self, other: Tensor<B, D>) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_add(self.primitive, other.primitive))
    }

    pub fn mul(self, other: Tensor<B, D>) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_mul(self.primitive, other.primitive))
    }

    pub fn sub(self, other: Tensor<B, D>) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_sub(self.primitive, other.primitive))
    }

    pub fn div(self, other: Tensor<B, D>) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_div(self.primitive, other.primitive))
    }

    pub fn pow(self, other: Tensor<B, D>) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_pow(self.primitive, other.primitive))
    }

    pub fn pow_scalar(self, exponent: f64) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_pow_scalar(self.primitive, exponent))
    }

    pub fn neg(self) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_neg(self.primitive))
    }

    pub fn exp(self) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_exp(self.primitive))
    }

    pub fn log(self) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_log(self.primitive))
    }

    pub fn relu(self) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_relu(self.primitive))
    }

    pub fn sum(self) -> Tensor<B, 1> {
        Tensor::from_primitive(B::float_sum(self.primitive))
    }

    pub fn add_scalar(self, value: B::FloatElem) -> Tensor<B, D> {
        let device = Default::default();
        let scalar = Self::scalar(&device, value);
        self.add(scalar)
    }

    pub fn mul_scalar(self, value: B::FloatElem) -> Tensor<B, D> {
        let device = Default::default();
        let scalar = Self::scalar(&device, value);
        self.mul(scalar)
    }

    pub fn reshape<const D2: usize>(self, shape: Shape) -> Tensor<B, D2> {
        Tensor::from_primitive(B::float_reshape(self.primitive, shape))
    }

    pub fn broadcast_to(self, shape: Shape) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_broadcast_to(self.primitive, shape))
    }

    pub fn softmax(self, dim: usize) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_softmax(self.primitive, dim))
    }

    pub fn matmul(self, other: Tensor<B, D>) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_matmul(self.primitive, other.primitive))
    }

    pub fn transpose(self) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_transpose(self.primitive))
    }

    pub fn sum_dim(self, dim: usize) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_sum_dim(self.primitive, dim))
    }

    pub fn mean_dim(self, dim: usize) -> Tensor<B, D> {
        Tensor::from_primitive(B::float_mean_dim(self.primitive, dim))
    }
}
