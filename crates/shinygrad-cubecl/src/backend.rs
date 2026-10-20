use std::marker::PhantomData;
use cubecl::bytes::Bytes;
use cubecl::Runtime;
use shinygrad_backend::{
    Backend, BackendTypes,
    FloatElement, FloatOps,
    Shape, TensorData,
    tensor::FloatTensor,
};

use crate::tensor::CubeTensorPrimitive;

pub struct CubeBackend<R: Runtime, E: FloatElement> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: FloatElement> CubeBackend<R, E> {
    pub fn new() -> Self { Self { _runtime: PhantomData, _elem: PhantomData } }
}

impl<R: Runtime, E: FloatElement> Default for CubeBackend<R, E> { fn default() -> Self { Self::new() } }
impl<R: Runtime, E: FloatElement> Clone for CubeBackend<R, E> { fn clone(&self) -> Self { Self::new() } }
impl<R: Runtime, E: FloatElement> core::fmt::Debug for CubeBackend<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.write_str("CubeBackend") }
}

fn elems_to_bytes<E>(elems: &[E]) -> Bytes {
    let size = std::mem::size_of::<E>();
    let ptr = elems.as_ptr() as *const u8;
    let raw = unsafe { std::slice::from_raw_parts(ptr, elems.len() * size) };
    Bytes::from_bytes_vec(raw.to_vec())
}

fn read_as_elems<E: FloatElement>(client: &cubecl::client::ComputeClient<impl Runtime>, handle: cubecl::server::Handle, n: usize) -> Vec<E> {
    let bytes = client.read_one(handle).unwrap();
    let raw: Vec<u8> = bytes.to_vec();
    let ptr = raw.as_ptr() as *const E;
    let len = raw.len() / std::mem::size_of::<E>();
    assert!(len >= n, "expected {n} elements, got {len}");
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

fn read_two<E: FloatElement>(client: &cubecl::client::ComputeClient<impl Runtime>, lh: cubecl::server::Handle, ln: usize, rh: cubecl::server::Handle, rn: usize, out_n: usize) -> (Vec<E>, Vec<E>) {
    let lv = read_as_elems::<E>(client, lh, ln);
    let rv = read_as_elems::<E>(client, rh, rn);
    if ln == rn { return (lv, rv); }
    if ln == 1 { return (vec![lv[0]; out_n], rv); }
    if rn == 1 { return (lv, vec![rv[0]; out_n]); }
    panic!("cannot broadcast {ln} vs {rn}");
}

impl<R, E> BackendTypes for CubeBackend<R, E>
where R: Runtime, E: FloatElement, R::Device: Default
{ type Device = R::Device; type FloatTensorPrimitive = CubeTensorPrimitive<R, E>; type FloatElem = E; }

impl<R, E> Backend for CubeBackend<R, E>
where R: Runtime, E: FloatElement, R::Device: Default
{ fn name(_: &Self::Device) -> String { format!("cubecl-{}", std::any::type_name::<E>()) } fn seed(_: &Self::Device, _: u64) {} }

impl<R, E> FloatOps<CubeBackend<R, E>> for CubeBackend<R, E>
where R: Runtime, E: FloatElement, R::Device: Default
{
    fn float_zeros(device: &R::Device, shape: Shape) -> FloatTensor<Self> {
        let client = R::client(device);
        let n = shape.num_elements();
        let h = client.create(elems_to_bytes::<E>(&vec![E::from_f64(0.0); n]));
        CubeTensorPrimitive::new_contiguous(client, device.clone(), shape, h)
    }
    fn float_ones(device: &R::Device, shape: Shape) -> FloatTensor<Self> {
        let client = R::client(device);
        let n = shape.num_elements();
        let h = client.create(elems_to_bytes::<E>(&vec![E::from_f64(1.0); n]));
        CubeTensorPrimitive::new_contiguous(client, device.clone(), shape, h)
    }
    fn float_from_data(device: &R::Device, data: TensorData<E>) -> FloatTensor<Self> {
        let client = R::client(device);
        let shape = data.shape.clone();
        let h = client.create(elems_to_bytes::<E>(&data.value));
        CubeTensorPrimitive::new_contiguous(client, device.clone(), shape, h)
    }
    fn float_into_data(tensor: FloatTensor<Self>) -> TensorData<E> {
        let n = tensor.shape.num_elements();
        let values = read_as_elems::<E>(&tensor.client, tensor.handle.clone(), n);
        TensorData::new(values, tensor.shape.clone())
    }
    fn float_shape(value: &FloatTensor<Self>) -> Shape { value.shape.clone() }
    fn float_scalar(device: &R::Device, value: E) -> FloatTensor<Self> {
        let client = R::client(device);
        let h = client.create(elems_to_bytes::<E>(&[value]));
        CubeTensorPrimitive::new_contiguous(client, device.clone(), Shape::new(vec![1]), h)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = lhs.client.clone(); let d = lhs.device.clone(); let s = lhs.shape.clone();
        let n = s.num_elements();
        let (lv, rv) = read_two::<E>(&c, lhs.handle.clone(), n, rhs.handle.clone(), rhs.shape.num_elements(), n);
        let out: Vec<E> = lv.iter().zip(&rv).map(|(a,b)| E::from_f64(a.to_f64()+b.to_f64())).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = lhs.client.clone(); let d = lhs.device.clone(); let s = lhs.shape.clone();
        let n = s.num_elements();
        let (lv, rv) = read_two::<E>(&c, lhs.handle.clone(), n, rhs.handle.clone(), rhs.shape.num_elements(), n);
        let out: Vec<E> = lv.iter().zip(&rv).map(|(a,b)| E::from_f64(a.to_f64()*b.to_f64())).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = lhs.client.clone(); let d = lhs.device.clone(); let s = lhs.shape.clone();
        let n = s.num_elements();
        let (lv, rv) = read_two::<E>(&c, lhs.handle.clone(), n, rhs.handle.clone(), rhs.shape.num_elements(), n);
        let out: Vec<E> = lv.iter().zip(&rv).map(|(a,b)| E::from_f64(a.to_f64()-b.to_f64())).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = lhs.client.clone(); let d = lhs.device.clone(); let s = lhs.shape.clone();
        let n = s.num_elements();
        let (lv, rv) = read_two::<E>(&c, lhs.handle.clone(), n, rhs.handle.clone(), rhs.shape.num_elements(), n);
        let out: Vec<E> = lv.iter().zip(&rv).map(|(a,b)| E::from_f64(a.to_f64()/b.to_f64())).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_pow_scalar(lhs: FloatTensor<Self>, exponent: f64) -> FloatTensor<Self> {
        let c = lhs.client.clone(); let d = lhs.device.clone(); let s = lhs.shape.clone();
        let n = s.num_elements();
        let lv = read_as_elems::<E>(&c, lhs.handle.clone(), n);
        let out: Vec<E> = lv.iter().map(|a| E::from_f64(a.to_f64().powf(exponent))).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_sum(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone();
        let n = value.shape.num_elements();
        let dv = read_as_elems::<E>(&c, value.handle.clone(), n);
        let sum: f64 = dv.iter().map(|v| v.to_f64()).sum();
        let h = c.create(elems_to_bytes::<E>(&[E::from_f64(sum)]));
        CubeTensorPrimitive::new_contiguous(c, d, Shape::new(vec![1]), h)
    }
    fn float_neg(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone(); let s = value.shape.clone();
        let n = s.num_elements();
        let v = read_as_elems::<E>(&c, value.handle.clone(), n);
        let out: Vec<E> = v.iter().map(|a| E::from_f64(-a.to_f64())).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_exp(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone(); let s = value.shape.clone();
        let n = s.num_elements();
        let v = read_as_elems::<E>(&c, value.handle.clone(), n);
        let out: Vec<E> = v.iter().map(|a| E::from_f64(a.to_f64().exp())).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_log(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone(); let s = value.shape.clone();
        let n = s.num_elements();
        let v = read_as_elems::<E>(&c, value.handle.clone(), n);
        let out: Vec<E> = v.iter().map(|a| E::from_f64(a.to_f64().ln())).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_relu(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone(); let s = value.shape.clone();
        let n = s.num_elements();
        let v = read_as_elems::<E>(&c, value.handle.clone(), n);
        let out: Vec<E> = v.iter().map(|a| { let x = a.to_f64(); E::from_f64(if x > 0.0 { x } else { 0.0 }) }).collect();
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, s, h)
    }
    fn float_sum_dim(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone(); let shape = value.shape.clone();
        let n = shape.num_elements();
        let v = read_as_elems::<E>(&c, value.handle.clone(), n);
        let mut out_shape = shape.dims.clone(); out_shape[dim] = 1;
        let out_n: usize = out_shape.iter().product();
        let mut out = vec![E::from_f64(0.0); out_n];
        if shape.dims.len() == 2 {
            let (r, col) = (shape.dims[0], shape.dims[1]);
            if dim == 0 { for j in 0..col { let mut s = 0.0f64; for i in 0..r { s += v[i*col+j].to_f64(); } out[j] = E::from_f64(s); } }
            else { for i in 0..r { let mut s = 0.0f64; for j in 0..col { s += v[i*col+j].to_f64(); } out[i] = E::from_f64(s); } }
        } else { out[0] = E::from_f64(v.iter().map(|x| x.to_f64()).sum()); }
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, Shape::new(out_shape), h)
    }
    fn float_mean_dim(value: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let n = value.shape.dims[dim];
        let summed = Self::float_sum_dim(value, dim);
        let sv = read_as_elems::<E>(&summed.client, summed.handle.clone(), summed.shape.num_elements());
        let out: Vec<E> = sv.iter().map(|a| E::from_f64(a.to_f64() / n as f64)).collect();
        let h = summed.client.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(summed.client.clone(), summed.device, summed.shape, h)
    }
    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = lhs.client.clone(); let d = lhs.device.clone();
        let (m, k) = (lhs.shape.dims[0], lhs.shape.dims[1]);
        let (k2, n) = (rhs.shape.dims[0], rhs.shape.dims[1]);
        assert_eq!(k, k2);
        let lv = read_as_elems::<E>(&c, lhs.handle.clone(), m*k);
        let rv = read_as_elems::<E>(&c, rhs.handle.clone(), k*n);
        let mut out = vec![E::from_f64(0.0); m*n];
        for i in 0..m { for j in 0..n { let mut s = 0.0f64; for p in 0..k { s += lv[i*k+p].to_f64() * rv[p*n+j].to_f64(); } out[i*n+j] = E::from_f64(s); } }
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, Shape::new(vec![m, n]), h)
    }
    fn float_transpose(value: FloatTensor<Self>) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone();
        let (rows, cols) = (value.shape.dims[0], value.shape.dims[1]);
        let v = read_as_elems::<E>(&c, value.handle.clone(), rows*cols);
        let mut out = vec![E::from_f64(0.0); rows*cols];
        for i in 0..rows { for j in 0..cols { out[j*rows+i] = v[i*cols+j]; } }
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, Shape::new(vec![cols, rows]), h)
    }
    fn float_reshape(value: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        CubeTensorPrimitive::new_contiguous(value.client.clone(), value.device.clone(), shape, value.handle.clone())
    }
    fn float_broadcast_to(value: FloatTensor<Self>, target_shape: Shape) -> FloatTensor<Self> {
        let c = value.client.clone(); let d = value.device.clone();
        let src_n = value.shape.num_elements();
        let tgt_n = target_shape.num_elements();
        let v = read_as_elems::<E>(&c, value.handle.clone(), src_n);
        let mut out = vec![E::from_f64(0.0); tgt_n];
        if src_n == 1 { for i in 0..tgt_n { out[i] = v[0]; } }
        else if value.shape.dims.len() == 2 && target_shape.dims.len() == 2 {
            let (sr, sc) = (value.shape.dims[0], value.shape.dims[1]);
            let (tr, tc) = (target_shape.dims[0], target_shape.dims[1]);
            for i in 0..tr { for j in 0..tc { let si = if sr == 1 { 0 } else { i }; let sj = if sc == 1 { 0 } else { j }; out[i*tc+j] = v[si*sc+sj]; } }
        } else { for i in 0..tgt_n { out[i] = v[i % src_n]; } }
        let h = c.create(elems_to_bytes::<E>(&out));
        CubeTensorPrimitive::new_contiguous(c, d, target_shape, h)
    }

    fn float_pow(_: FloatTensor<Self>, _: FloatTensor<Self>) -> FloatTensor<Self> { todo!() }
    fn float_softmax(_: FloatTensor<Self>, _: usize) -> FloatTensor<Self> { todo!() }
}
