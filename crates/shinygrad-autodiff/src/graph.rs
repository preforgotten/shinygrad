use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use shinygrad_backend::Backend;

use crate::tensor::AutodiffTensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

pub struct Node<B: Backend> {
    pub id: NodeId,
    pub op: String,
    pub inputs: Vec<Arc<Node<B>>>,
    pub backward: RefCell<Option<Box<dyn Fn() + 'static>>>,
    pub grad: RefCell<<B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive>,
}

impl<B: Backend> Node<B> {
    pub fn leaf(
        id: NodeId,
        op: &str,
        zero: <B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive,
    ) -> Self {
        Node {
            id,
            op: op.to_string(),
            inputs: vec![],
            backward: RefCell::new(None),
            grad: RefCell::new(zero),
        }
    }
}

#[derive(Clone)]
pub struct Gradients<B: Backend> {
    map: std::rc::Rc<RefCell<HashMap<NodeId, <B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive>>>,
}

impl<B: Backend> Gradients<B> {
    pub fn new() -> Self {
        Gradients {
            map: std::rc::Rc::new(RefCell::new(HashMap::new())),
        }
    }

    pub fn get(
        &self,
        node_id: NodeId,
    ) -> Option<<B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive> {
        self.map.borrow().get(&node_id).cloned()
    }

    pub fn remove(
        &self,
        node_id: NodeId,
    ) -> Option<<B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive> {
        self.map.borrow_mut().remove(&node_id)
    }

    pub fn insert(
        &self,
        node_id: NodeId,
        grad: <B as shinygrad_backend::BackendTypes>::FloatTensorPrimitive,
    ) {
        self.map.borrow_mut().insert(node_id, grad);
    }
}

fn build_topo<B: Backend>(
    node: &Arc<Node<B>>,
    topo: &mut Vec<Arc<Node<B>>>,
    visited: &mut HashSet<NodeId>,
) {
    if visited.contains(&node.id) {
        return;
    }
    visited.insert(node.id);
    for input in &node.inputs {
        build_topo(input, topo, visited);
    }
    topo.push(Arc::clone(node));
}

pub fn backward<B: Backend>(root: &AutodiffTensor<B>) -> Gradients<B> {
    let mut topo: Vec<Arc<Node<B>>> = vec![];
    let mut visited: HashSet<NodeId> = HashSet::new();
    build_topo(&root.node, &mut topo, &mut visited);

    let shape = B::float_shape(&root.primitive);
    let device: <B as shinygrad_backend::BackendTypes>::Device = Default::default();
    *root.node.grad.borrow_mut() = B::float_ones(&device, shape);

    for node in topo.iter().rev() {
        if let Some(ref backward_fn) = *node.backward.borrow() {
            backward_fn();
        }
    }

    let grads = Gradients::new();
    for node in &topo {
        let grad = node.grad.borrow().clone();
        grads.insert(node.id, grad);
    }

    grads
}
