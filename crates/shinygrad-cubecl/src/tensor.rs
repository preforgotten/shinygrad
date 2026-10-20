use std::marker::PhantomData;
use cubecl::client::ComputeClient;
use cubecl::server::Handle;
use cubecl::Runtime;
use shinygrad_backend::{FloatElement, Shape};

/// CubeCL tensor primitive — wraps a server handle with shape metadata.
pub struct CubeTensorPrimitive<R: Runtime, E: FloatElement> {
    pub client: ComputeClient<R>,
    pub handle: Handle,
    pub shape: Shape,
    pub device: R::Device,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: FloatElement> CubeTensorPrimitive<R, E> {
    pub fn new_contiguous(
        client: ComputeClient<R>,
        device: R::Device,
        shape: Shape,
        handle: Handle,
    ) -> Self {
        Self { client, handle, shape, device, _elem: PhantomData }
    }
}

impl<R: Runtime, E: FloatElement> Clone for CubeTensorPrimitive<R, E> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            _elem: PhantomData,
        }
    }
}

impl<R: Runtime, E: FloatElement> core::fmt::Debug for CubeTensorPrimitive<R, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CubeTensorPrimitive")
            .field("shape", &self.shape)
            .field("device", &self.device)
            .finish()
    }
}
