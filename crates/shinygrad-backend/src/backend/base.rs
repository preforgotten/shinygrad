use crate::element::FloatElement;
use crate::tensor::FloatTensor;

use super::ops::float::FloatOps;

pub trait BackendTypes {
    type Device: Clone + Send + Sync + core::fmt::Debug + Default + 'static;
    type FloatTensorPrimitive: Clone + core::fmt::Debug + 'static;
    type FloatElem: FloatElement;
}

pub trait Backend:
    BackendTypes
    + FloatOps<Self>
    + Clone
    + Default
    + Sized
    + core::fmt::Debug
    + 'static
{
    fn name(device: &Self::Device) -> String;
    fn seed(device: &<Self as BackendTypes>::Device, seed: u64);

    fn sync(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }
}

pub trait AutodiffBackend: Backend {
    type InnerBackend: Backend<Device = Self::Device, FloatElem = Self::FloatElem>;
    type Gradients: Clone;

    fn backward(tensor: FloatTensor<Self>) -> Self::Gradients;

    fn grad(
        tensor: &FloatTensor<Self>,
        grads: &Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>>;

    fn grad_remove(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>>;

    fn grad_replace(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
        grad: FloatTensor<Self::InnerBackend>,
    );

    fn inner(tensor: FloatTensor<Self>) -> FloatTensor<Self::InnerBackend>;

    fn from_inner(tensor: FloatTensor<Self::InnerBackend>) -> FloatTensor<Self>;
}

#[derive(Debug, Clone)]
pub enum ExecutionError {
    Generic(String),
}

impl core::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExecutionError::Generic(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ExecutionError {}
