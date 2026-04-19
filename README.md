# AMD Heterogeneous Programming Learning Kit

Learn heterogeneous programming on your **AMD Ryzen AI 7 PRO 350** system,
which has three compute units: CPU (Zen 5), GPU (RDNA 3.5), and NPU (XDNA 2).

## Your Hardware

| Component | Details |
|-----------|---------|
| **CPU** | AMD Ryzen AI 7 PRO 350 — 8 cores / 16 threads (Zen 5) |
| **GPU** | AMD Radeon 860M — RDNA 3.5 integrated, ~4 GB shared |
| **NPU** | AMD XDNA 2 — up to 50 TOPS (INT8), AI accelerator |

## Quick Start

```powershell
# One-command setup
.\setup_env.ps1

# Or manually:
uv sync
uv run python src/01_system_overview.py
```

## Sample Scripts

| Script | Description |
|--------|-------------|
| `src/01_system_overview.py` | Detects CPU, GPU, NPU and prints a full system report |
| `src/02_gpu_config.py` | GPU deep-dive: OpenCL properties, DirectML check, compute benchmark |
| `src/03_npu_config.py` | NPU deep-dive: driver status, ONNX Runtime providers, architecture overview |
| `src/04_heterogeneous_demo.py` | Benchmarks the same workload on CPU vs GPU vs NPU |

## Programming Models

### CPU (Zen 5)
- **NumPy** for vectorized math
- **ONNX Runtime** `CPUExecutionProvider`
- Standard Python multiprocessing/threading

### GPU (Radeon 860M)
- **OpenCL** via `pyopencl` — write custom GPU kernels
- **DirectML** via `onnxruntime-directml` — accelerated ML inference
- Vulkan Compute (via Vulkan SDK, not covered here)

### NPU (XDNA 2)
- **ONNX Runtime** with `VitisAIExecutionProvider`
- Requires [AMD Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/)
- Best for INT8/INT4 quantized models

## Enabling NPU Acceleration

The NPU requires the AMD Ryzen AI Software SDK:

1. Download from https://ryzenai.docs.amd.com/en/latest/inst.html
2. Install the NPU driver + runtime
3. Install the Python EP: `pip install vitis-ai-execution-provider`
4. Set: `$env:XLNX_VART_FIRMWARE = "C:\path\to\xclbin"`

## Dependencies

Managed via `uv` (see `pyproject.toml`):
- `numpy` — array computing
- `pyopencl` — OpenCL GPU compute
- `onnxruntime-directml` — ML inference on CPU + GPU (DirectML)
- `psutil`, `wmi`, `tabulate` — system detection and formatting
