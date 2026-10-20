use crate::element::FloatElement;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct TensorData<E: FloatElement> {
    pub value: Vec<E>,
    pub shape: Shape,
}

impl<E: FloatElement> TensorData<E> {
    pub fn new(value: Vec<E>, shape: Shape) -> Self {
        assert_eq!(
            value.len(),
            shape.num_elements(),
            "TensorData value length {} does not match shape {}",
            value.len(),
            shape.num_elements()
        );
        TensorData { value, shape }
    }

    pub fn from_scalar(value: E) -> Self {
        TensorData {
            value: vec![value],
            shape: Shape::new(vec![1]),
        }
    }
}
