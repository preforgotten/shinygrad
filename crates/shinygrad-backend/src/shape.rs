use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    pub fn from<const D: usize>(dims: [usize; D]) -> Self {
        Shape {
            dims: dims.to_vec(),
        }
    }

    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    }
}
