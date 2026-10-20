use crate::backend::BackendTypes;

pub type FloatTensor<B> = <B as BackendTypes>::FloatTensorPrimitive;
