use std::marker::PhantomData;
use std::sync::Arc;
use std::cell::RefCell;

use shinygrad_backend::{
    AutodiffBackend as AutodiffBackendTrait,
    Backend,
    BackendTypes,
    ExecutionError,
    FloatElement,
    FloatOps,
    Shape,
    TensorData,
    tensor::FloatTensor,
};

use crate::graph::{Node, backward as run_backward, Gradients};
use crate::tensor::AutodiffTensor;

#[derive(Debug, Clone, Copy, Default)]
pub struct Autodiff<B: Backend> {
    _b: PhantomData<B>,
}

impl<B: Backend> Autodiff<B> {
    pub fn new() -> Self {
        Autodiff { _b: PhantomData }
    }
}

impl<B: Backend> BackendTypes for Autodiff<B> {
    type Device = <B as BackendTypes>::Device;
    type FloatTensorPrimitive = AutodiffTensor<B>;
    type FloatElem = <B as BackendTypes>::FloatElem;
}

impl<B: Backend> Backend for Autodiff<B> {
    fn name(device: &Self::Device) -> String {
        format!("autodiff<{}>", B::name(device))
    }

    fn seed(device: &<Self as BackendTypes>::Device, seed: u64) {
        B::seed(device, seed)
    }

    fn sync(device: &<Self as BackendTypes>::Device) -> Result<(), ExecutionError> {
        B::sync(device)
    }
}

impl<B: Backend> AutodiffBackendTrait for Autodiff<B> {
    type InnerBackend = B;
    type Gradients = Gradients<B>;

    fn backward(tensor: FloatTensor<Self>) -> Self::Gradients {
        run_backward(&tensor)
    }

    fn grad(
        tensor: &FloatTensor<Self>,
        grads: &Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>> {
        grads.get(tensor.node.id)
    }

    fn grad_remove(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
    ) -> Option<FloatTensor<Self::InnerBackend>> {
        grads.remove(tensor.node.id)
    }

    fn grad_replace(
        tensor: &FloatTensor<Self>,
        grads: &mut Self::Gradients,
        grad: FloatTensor<Self::InnerBackend>,
    ) {
        grads.insert(tensor.node.id, grad);
    }

    fn inner(tensor: FloatTensor<Self>) -> FloatTensor<Self::InnerBackend> {
        tensor.primitive
    }

    fn from_inner(tensor: FloatTensor<Self::InnerBackend>) -> FloatTensor<Self> {
        AutodiffTensor::new(tensor)
    }
}

impl<B: Backend> FloatOps<Autodiff<B>> for Autodiff<B> {
    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_add(lhs.primitive.clone(), rhs.primitive.clone());
        let lhs_n = lhs.node.clone();
        let rhs_n = rhs.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "add".into(),
            inputs: vec![lhs_n.clone(), rhs_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _old = lhs_n.grad.borrow().clone(); let _new = B::float_add(_old, grad_out.clone()); *lhs_n.grad.borrow_mut() = _new;
            let _old = rhs_n.grad.borrow().clone(); let _new = B::float_add(_old, grad_out); *rhs_n.grad.borrow_mut() = _new;
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_mul(lhs.primitive.clone(), rhs.primitive.clone());
        let lhs_data_inner = lhs.primitive.clone();
        let rhs_data_inner = rhs.primitive.clone();
        let lhs_n = lhs.node.clone();
        let rhs_n = rhs.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "mul".into(),
            inputs: vec![lhs_n.clone(), rhs_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _grad_lhs_n = lhs_n.grad.borrow().clone();

            *lhs_n.grad.borrow_mut() = B::float_add(
                _grad_lhs_n.clone(),
                B::float_mul(grad_out.clone(), rhs_data_inner.clone()),
            );
            let _grad_rhs_n = rhs_n.grad.borrow().clone();

            *rhs_n.grad.borrow_mut() = B::float_add(
                _grad_rhs_n.clone(),
                B::float_mul(grad_out, lhs_data_inner.clone()),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_sub(lhs.primitive.clone(), rhs.primitive.clone());
        let lhs_n = lhs.node.clone();
        let rhs_n = rhs.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "sub".into(),
            inputs: vec![lhs_n.clone(), rhs_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _old = lhs_n.grad.borrow().clone(); let _new = B::float_add(_old, grad_out.clone()); *lhs_n.grad.borrow_mut() = _new;
            let _old = rhs_n.grad.borrow().clone(); let _new = B::float_add(_old, B::float_neg(grad_out)); *rhs_n.grad.borrow_mut() = _new;
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_div(lhs.primitive.clone(), rhs.primitive.clone());
        let lhs_data = lhs.primitive.clone();
        let rhs_data = rhs.primitive.clone();
        let lhs_n = lhs.node.clone();
        let rhs_n = rhs.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "div".into(),
            inputs: vec![lhs_n.clone(), rhs_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _grad_lhs_n = lhs_n.grad.borrow().clone();

            *lhs_n.grad.borrow_mut() = B::float_add(
                _grad_lhs_n.clone(),
                B::float_div(grad_out.clone(), rhs_data.clone()),
            );
            let neg_lhs = B::float_neg(lhs_data.clone());
            let rhs_sq = B::float_mul(rhs_data.clone(), rhs_data.clone());
            let _grad_rhs_n = rhs_n.grad.borrow().clone();

            *rhs_n.grad.borrow_mut() = B::float_add(
                _grad_rhs_n.clone(),
                B::float_div(B::float_mul(grad_out, neg_lhs), rhs_sq),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_pow(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_pow(lhs.primitive.clone(), rhs.primitive.clone());
        let lhs_data = lhs.primitive.clone();
        let rhs_data = rhs.primitive.clone();
        let inner_clone = inner.clone();
        let lhs_n = lhs.node.clone();
        let rhs_n = rhs.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "pow".into(),
            inputs: vec![lhs_n.clone(), rhs_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let rhs_minus_one = B::float_sub(
                rhs_data.clone(),
                B::float_ones(&Default::default(), B::float_shape(&rhs_data)),
            );
            let lhs_pow = B::float_pow(lhs_data.clone(), rhs_minus_one);
            let _grad_lhs_n = lhs_n.grad.borrow().clone();

            *lhs_n.grad.borrow_mut() = B::float_add(
                _grad_lhs_n.clone(),
                B::float_mul(B::float_mul(grad_out.clone(), rhs_data.clone()), lhs_pow),
            );
            let _grad_rhs_n = rhs_n.grad.borrow().clone();

            *rhs_n.grad.borrow_mut() = B::float_add(
                _grad_rhs_n.clone(),
                B::float_mul(B::float_mul(grad_out, inner_clone.clone()), B::float_log(lhs_data.clone())),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_pow_scalar(lhs: FloatTensor<Self>, exponent: f64) -> FloatTensor<Self> {
        let inner = B::float_pow_scalar(lhs.primitive.clone(), exponent);
        let lhs_data = lhs.primitive.clone();
        let lhs_n = lhs.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "pow_scalar".into(),
            inputs: vec![lhs_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let e_minus_one = exponent - 1.0;
            let scaled = B::float_pow_scalar(lhs_data.clone(), e_minus_one);
            let scaled = B::float_mul(grad_out, scaled);
            let e_scalar = B::float_scalar(&Default::default(), <B::FloatElem as FloatElement>::from_f64(exponent));
            let scaled = B::float_mul(scaled, e_scalar);
            let _old = lhs_n.grad.borrow().clone();
            *lhs_n.grad.borrow_mut() = B::float_add(_old, scaled);
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_neg(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_neg(value.primitive.clone());
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "neg".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _old = v_n.grad.borrow().clone(); let _new = B::float_add(_old, B::float_neg(grad_out)); *v_n.grad.borrow_mut() = _new;
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_exp(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_exp(value.primitive.clone());
        let out_data = inner.clone();
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "exp".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _grad_v_n = v_n.grad.borrow().clone();

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                B::float_mul(grad_out, out_data.clone()),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_log(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_log(value.primitive.clone());
        let val_data = value.primitive.clone();
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "log".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _grad_v_n = v_n.grad.borrow().clone();

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                B::float_div(grad_out, val_data.clone()),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_relu(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_relu(value.primitive.clone());
        let val_data = value.primitive.clone();
        let relu_out = inner.clone();
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "relu".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _grad_v_n = v_n.grad.borrow().clone();
            let eps = B::float_scalar(&Default::default(), <B::FloatElem as FloatElement>::from_f64(1e-10));
            let safe_denom = B::float_add(val_data.clone(), eps);

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                B::float_mul(grad_out, B::float_div(relu_out.clone(), safe_denom)),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_sum(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_sum(value.primitive.clone());
        let v_shape = B::float_shape(&value.primitive);
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "sum".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let grad_broadcast = B::float_broadcast_to(grad_out, v_shape.clone());
            let _grad_v_n = v_n.grad.borrow().clone();

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                grad_broadcast,
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_sum_dim(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let inner = B::float_sum_dim(value.primitive.clone(), dim);
        let v_shape = B::float_shape(&value.primitive);
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "sum_dim".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let grad_broadcast = B::float_broadcast_to(grad_out, v_shape.clone());
            let _grad_v_n = v_n.grad.borrow().clone();

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                grad_broadcast,
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_mean_dim(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let inner = B::float_mean_dim(value.primitive.clone(), dim);
        let v_shape = B::float_shape(&value.primitive);
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "mean_dim".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let grad_broadcast = B::float_broadcast_to(grad_out, v_shape.clone());
            let _grad_v_n = v_n.grad.borrow().clone();

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                grad_broadcast,
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_matmul(lhs.primitive.clone(), rhs.primitive.clone());
        let lhs_data = lhs.primitive.clone();
        let rhs_data = rhs.primitive.clone();
        let lhs_n = lhs.node.clone();
        let rhs_n = rhs.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "matmul".into(),
            inputs: vec![lhs_n.clone(), rhs_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let rhs_t = B::float_transpose(rhs_data.clone());
            let _grad_lhs_n = lhs_n.grad.borrow().clone();

            *lhs_n.grad.borrow_mut() = B::float_add(
                _grad_lhs_n.clone(),
                B::float_matmul(grad_out.clone(), rhs_t),
            );
            let lhs_t = B::float_transpose(lhs_data.clone());
            let _grad_rhs_n = rhs_n.grad.borrow().clone();

            *rhs_n.grad.borrow_mut() = B::float_add(
                _grad_rhs_n.clone(),
                B::float_matmul(lhs_t, grad_out),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_transpose(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let inner = B::float_transpose(value.primitive.clone());
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "transpose".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _old = v_n.grad.borrow().clone(); let _new = B::float_add(_old, B::float_transpose(grad_out)); *v_n.grad.borrow_mut() = _new;
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_softmax(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let inner = B::float_softmax(value.primitive.clone(), dim);
        let softmax_out = inner.clone();
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "softmax".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let s_times_grad = B::float_mul(softmax_out.clone(), grad_out.clone());
            let sum_sg = B::float_sum_dim(s_times_grad, dim);
            let grad_sub = B::float_sub(grad_out, sum_sg);
            let _grad_v_n = v_n.grad.borrow().clone();

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                B::float_mul(softmax_out.clone(), grad_sub),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_reshape(value: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let inner = B::float_reshape(value.primitive.clone(), shape);
        let orig_shape = B::float_shape(&value.primitive);
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "reshape".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            let _grad_v_n = v_n.grad.borrow().clone();

            *v_n.grad.borrow_mut() = B::float_add(
                _grad_v_n.clone(),
                B::float_reshape(grad_out, orig_shape.clone()),
            );
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_broadcast_to(value: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let inner = B::float_broadcast_to(value.primitive.clone(), shape.clone());
        let orig_shape = B::float_shape(&value.primitive);
        let new_shape = shape.clone();
        let v_n = value.node.clone();
        let out_node = Arc::new(Node {
            id: crate::tensor::next_id(),
            op: "broadcast_to".into(),
            inputs: vec![v_n.clone()],
            backward: RefCell::new(None),
            grad: RefCell::new(B::float_zeros(&Default::default(), B::float_shape(&inner))),
        });
        let out_n = out_node.clone();
        *out_node.backward.borrow_mut() = Some(Box::new(move || {
            let grad_out = out_n.grad.borrow().clone();
            // Sum grad along dimensions that were broadcast (where orig had size 1, new has size > 1)
            let mut grad = grad_out;
            for d in (0..orig_shape.dims.len()).rev() {
                let orig_d = orig_shape.dims.get(d).copied().unwrap_or(1);
                let new_d = new_shape.dims.get(d).copied().unwrap_or(1);
                if orig_d == 1 && new_d > 1 {
                    grad = B::float_sum_dim(grad, d);
                }
            }
            // Reshape to original shape
            let grad_orig = B::float_reshape(grad, orig_shape.clone());
            let _grad_v_n = v_n.grad.borrow().clone();
            *v_n.grad.borrow_mut() = B::float_add(_grad_v_n.clone(), grad_orig);
        }));
        AutodiffTensor { primitive: inner, node: out_node }
    }

    fn float_shape(value: &FloatTensor<Self>) -> Shape {
        B::float_shape(&value.primitive)
    }

    fn float_into_data(value: FloatTensor<Self>) -> TensorData<<Self as BackendTypes>::FloatElem> {
        B::float_into_data(value.primitive)
    }

    fn float_zeros(device: &<Self as BackendTypes>::Device, shape: Shape) -> FloatTensor<Self> {
        let inner = B::float_zeros(device, shape.clone());
        AutodiffTensor::new(inner)
    }

    fn float_ones(device: &<Self as BackendTypes>::Device, shape: Shape) -> FloatTensor<Self> {
        let inner = B::float_ones(device, shape.clone());
        AutodiffTensor::new(inner)
    }

    fn float_from_data(device: &<Self as BackendTypes>::Device, data: TensorData<<Self as BackendTypes>::FloatElem>) -> FloatTensor<Self> {
        let inner = B::float_from_data(device, data);
        AutodiffTensor::new(inner)
    }
}
