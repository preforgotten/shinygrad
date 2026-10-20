# shinygrad

A modular deep learning framework in Rust, built from scratch for learning.
Architecture heavily inspired by [Burn](https://github.com/tracel-ai/burn).

## Crates

| Crate | Burn equivalent | Description |
|---|---|---|
| [`shinygrad-backend`](crates/shinygrad-backend/) | `burn-backend` | `Backend` trait, `FloatOps`, `DType`, `Shape`, `Device` |
| [`shinygrad-tensor`](crates/shinygrad-tensor/) | `burn-tensor` | `Tensor<B, D>` — user-facing tensor API |
| [`shinygrad-autodiff`](crates/shinygrad-autodiff/) | `burn-autodiff` | `Autodiff<B>` decorator, computation graph, backward pass |
| [`shinygrad-cpu-scratch`](crates/shinygrad-cpu-scratch/) | `burn-ndarray` | Handwritten CPU backend via [ndarray](https://crates.io/crates/ndarray) |
| [`shinygrad-core`](crates/shinygrad-core/) | `burn-core` | `Module` trait, `Param` |
| [`shinygrad-nn`](crates/shinygrad-nn/) | `burn-nn` | `Linear`, `ReLU`, `Sequential`, loss, initializers |
| [`shinygrad-optim`](crates/shinygrad-optim/) | `burn-optim` | `SGD`, `Adam`, `Momentum` |
| [`shinygrad`](crates/shinygrad/) | `burn` | Facade — re-exports everything |

## Quick start

```rust
use shinygrad::prelude::*;

type Inner = CpuBackend<f64>;
type MyBackend = Autodiff<Inner>;

let device = Device::Cpu;

let mut model: Sequential<MyBackend> = Sequential::new(vec![
    Box::new(LinearConfig::new(2, 4).init(&device)),
    Box::new(ReLU::new()),
    Box::new(LinearConfig::new(4, 2).init(&device)),
]);
```

## Examples

```bash
cargo run --example xor --release
cargo run --example mnist_mini --release
```

## Backend abstraction

All tensor operations go through the `Backend` trait. Adding a new backend
means implementing `Backend` + `FloatOps` — no model code changes needed.

| Backend | Status |
|---|---|
| `CpuBackend<f32>` / `CpuBackend<f64>` (scratch, ndarray) | Ready |
| CubeCL CPU backend | Planned |
| CubeCL GPU backend | Planned |

## Design

Shinygrad follows Burn's architecture:

```
User: Tensor<Autodiff<CpuBackend<f64>>, 2>
              │
              ▼
       Autodiff<B>     ← tracks computation graph, backward pass
              │
              ▼
       CpuBackend<E>   ← actual math on ndarray arrays
```

The `Autodiff<B>` decorator wraps any backend and adds gradient tracking
transparently. Non-autodiff backends can be used directly for inference.

## License

MIT OR Apache-2.0
