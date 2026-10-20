use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
        }
    }
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
}
