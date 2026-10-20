use std::sync::Arc;
use std::cell::RefCell;

use shinygrad_backend::Backend;

use crate::graph::{Node, NodeId};

static mut NEXT_NODE_ID: usize = 0;

pub fn next_id() -> NodeId {
    unsafe {
        let id = NEXT_NODE_ID;
        NEXT_NODE_ID += 1;
        NodeId(id)
    }
}

#[derive(Clone)]
pub struct AutodiffTensor<B: Backend> {
    pub primitive: <B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive,
    pub node: Arc<Node<B>>,
}

impl<B: Backend> AutodiffTensor<B> {
    pub fn new(
        primitive: <B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive,
    ) -> Self {
        let id = next_id();
        let shape = B::float_shape(&primitive);
        let device: <B as shinygrad_backend::BackendTypes>::Device = Default::default();
        let zero = B::float_zeros(&device, shape);
        let node = Arc::new(Node::leaf(id, "leaf", zero));
        AutodiffTensor { primitive, node }
    }

    pub fn from_op(
        primitive: <B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive,
        op: &str,
        inputs: Vec<Arc<Node<B>>>,
        backward_fn: Option<Box<dyn Fn() + 'static>>,
    ) -> Self {
        let id = next_id();
        let shape = B::float_shape(&primitive);
        let device: <B as shinygrad_backend::BackendTypes>::Device = Default::default();
        let zero = B::float_zeros(&device, shape);
        let node = Arc::new(Node {
            id,
            op: op.to_string(),
            inputs,
            backward: RefCell::new(backward_fn),
            grad: RefCell::new(zero),
        });
        AutodiffTensor { primitive, node }
    }
}

impl<B: Backend> core::fmt::Debug for AutodiffTensor<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AutodiffTensor")
            .field("id", &self.node.id)
            .field("op", &self.node.op)
            .finish()
    }
}
