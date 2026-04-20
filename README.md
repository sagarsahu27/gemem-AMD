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

The project uses the **`ryzen-ai-1.7.1`** conda environment (created by the AMD Ryzen AI SDK installer),
which includes all three execution providers: CPU, DirectML (GPU), and VitisAI (NPU).

```powershell
# Activate the Ryzen AI SDK environment
conda activate ryzen-ai-1.7.1

# Run the scripts
python src/01_system_overview.py
python src/02_gpu_config.py
python src/03_npu_config.py
python src/04_heterogeneous_demo.py
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

The NPU requires the AMD Ryzen AI Software SDK 1.7.1:

1. Download and install from https://ryzenai.docs.amd.com/en/latest/inst.html
2. The installer creates the `ryzen-ai-1.7.1` conda environment with all SDK wheels
3. Install additional packages for this project:
   ```powershell
   conda run -n ryzen-ai-1.7.1 pip install pyopencl wmi pywin32
   ```
4. Set firmware path (auto-detected by scripts, or set manually):
   ```powershell
   $env:XLNX_VART_FIRMWARE = "C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
   ```
5. Verify:
   ```powershell
   conda run -n ryzen-ai-1.7.1 python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   # Should show: ['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
   ```

## Dependencies

Managed via the **Ryzen AI SDK** conda environment `ryzen-ai-1.7.1`.

Additional packages installed for this project:
- `pyopencl` — OpenCL GPU compute
- `wmi`, `pywin32` — Windows system detection

Included in the SDK environment:
- `numpy` — array computing
- `onnxruntime-vitisai` — ML inference on CPU + GPU (DirectML) + NPU (VitisAI)
- `onnx` — model creation and manipulation
- `tabulate`, `psutil` — formatting and system info
