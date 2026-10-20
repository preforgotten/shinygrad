use crate::shape::Shape;
use crate::backend::BackendTypes;
use crate::tensor::FloatTensor;

pub trait FloatOps<B: BackendTypes> {
    fn float_add(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;
    fn float_mul(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;
    fn float_sub(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;
    fn float_div(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;
    fn float_pow(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;
    fn float_pow_scalar(lhs: FloatTensor<B>, exponent: f64) -> FloatTensor<B>;
    fn float_neg(value: FloatTensor<B>) -> FloatTensor<B>;
    fn float_exp(value: FloatTensor<B>) -> FloatTensor<B>;
    fn float_log(value: FloatTensor<B>) -> FloatTensor<B>;
    fn float_relu(value: FloatTensor<B>) -> FloatTensor<B>;
    fn float_sum(value: FloatTensor<B>) -> FloatTensor<B>;
    fn float_sum_dim(value: FloatTensor<B>, dim: usize) -> FloatTensor<B>;
    fn float_mean_dim(value: FloatTensor<B>, dim: usize) -> FloatTensor<B>;

    fn float_matmul(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;
    fn float_transpose(value: FloatTensor<B>) -> FloatTensor<B>;

    fn float_softmax(value: FloatTensor<B>, dim: usize) -> FloatTensor<B>;

    fn float_reshape(value: FloatTensor<B>, shape: Shape) -> FloatTensor<B>;

    fn float_broadcast_to(value: FloatTensor<B>, shape: Shape) -> FloatTensor<B>;

    fn float_shape(value: &FloatTensor<B>) -> Shape;

    fn float_into_data(value: FloatTensor<B>) -> crate::data::TensorData<B::FloatElem>;

    fn float_zeros(device: &B::Device, shape: Shape) -> FloatTensor<B>;
    fn float_ones(device: &B::Device, shape: Shape) -> FloatTensor<B>;
    fn float_from_data(
        device: &B::Device,
        data: crate::data::TensorData<B::FloatElem>,
    ) -> FloatTensor<B>;

    fn float_scalar(device: &B::Device, value: B::FloatElem) -> FloatTensor<B> {
        Self::float_from_data(device, crate::data::TensorData::from_scalar(value))
    }
}
