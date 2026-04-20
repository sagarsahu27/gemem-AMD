"""
AMD NPU Configuration Deep-Dive
=================================
Explores your AMD XDNA 2 Neural Processing Unit:
  1. NPU device detection and driver status
  2. ONNX Runtime execution provider check
  3. NPU capabilities and architecture overview
  4. Simple inference test (if NPU runtime is available)

Run: conda run -n ryzen-ai-1.7.1 python src/03_npu_config.py

NOTE: Full NPU acceleration requires the AMD Ryzen AI Software SDK.
      Download from: https://ryzenai.docs.amd.com/en/latest/
"""

import subprocess
import json
import sys


# ---------------------------------------------------------------------------
# 1. NPU Hardware Detection
# ---------------------------------------------------------------------------

def detect_npu_hardware() -> dict | None:
    """Detect the AMD NPU via Windows PnP device manager."""
    print("=" * 70)
    print("  NPU HARDWARE DETECTION")
    print("=" * 70)

    npu_info = None

    # Method 1: WMI
    try:
        import wmi
        c = wmi.WMI()
        for device in c.Win32_PnPEntity():
            name = device.Name or ""
            if "NPU" in name and "Compute Accelerator" in name:
                npu_info = {
                    "name": name.strip(),
                    "device_id": device.DeviceID,
                    "status": device.Status,
                    "manufacturer": device.Manufacturer or "AMD",
                    "pnp_class": device.PNPClass or "Unknown",
                }
                break
    except ImportError:
        pass

    # Method 2: PowerShell fallback
    if not npu_info:
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-PnpDevice -FriendlyName '*NPU Compute*' | "
                 "Select-Object FriendlyName, Status, InstanceId, Class | "
                 "ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]
            if data:
                d = data[0]
                npu_info = {
                    "name": d.get("FriendlyName", "").strip(),
                    "device_id": d.get("InstanceId", ""),
                    "status": "OK" if d.get("Status") == "OK" else d.get("Status", "Unknown"),
                    "manufacturer": "AMD",
                }
        except Exception:
            pass

    if npu_info:
        from tabulate import tabulate
        table = [
            ["Device Name", npu_info["name"]],
            ["Status", npu_info["status"]],
            ["Manufacturer", npu_info["manufacturer"]],
            ["PCI Device ID", npu_info["device_id"][:70]],
        ]
        print(tabulate(table, tablefmt="simple_grid"))

        # Parse the PCI Vendor/Device IDs
        dev_id = npu_info["device_id"]
        if "VEN_1022" in dev_id and "DEV_17F0" in dev_id:
            print("\n  Identified: AMD XDNA 2 (Strix Point) NPU")
            print("  Architecture: AMD XDNA 2")
            print("  AI Performance: up to 50 TOPS (INT8)")
            print("  Supported precisions: INT4, INT8, FP16, BF16")
    else:
        print("  No AMD NPU detected in this system.")

    return npu_info


# ---------------------------------------------------------------------------
# 2. NPU Driver Status
# ---------------------------------------------------------------------------

def check_npu_driver():
    """Check the AMD NPU driver installation status."""
    print("\n" + "=" * 70)
    print("  NPU DRIVER STATUS")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-PnpDevice -FriendlyName '*NPU Compute*' | "
             "Get-PnpDeviceProperty -KeyName 'DEVPKEY_Device_DriverVersion','DEVPKEY_Device_DriverDate',"
             "'DEVPKEY_Device_DriverProvider' | "
             "Select-Object KeyName, Data | ConvertTo-Json"],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]

        from tabulate import tabulate
        table = []
        for prop in data:
            key = prop.get("KeyName", "").split("_")[-1] if prop.get("KeyName") else "Unknown"
            value = prop.get("Data", "N/A")
            if isinstance(value, str) and len(value) > 80:
                value = value[:80] + "..."
            table.append([key, value])

        if table:
            print(tabulate(table, headers=["Property", "Value"], tablefmt="simple_grid"))
        else:
            print("  Could not retrieve driver properties.")
    except Exception as e:
        print(f"  Error querying driver: {e}")


# ---------------------------------------------------------------------------
# 3. ONNX Runtime Execution Providers
# ---------------------------------------------------------------------------

def check_onnxrt_providers():
    """Check which ONNX Runtime execution providers are available."""
    print("\n" + "=" * 70)
    print("  ONNX RUNTIME EXECUTION PROVIDERS")
    print("=" * 70)

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()

        print(f"\n  ONNX Runtime version: {ort.__version__}")
        print(f"  Available providers ({len(providers)}):\n")

        provider_info = {
            "CPUExecutionProvider": ("CPU", "Default fallback, runs on Zen 5 cores"),
            "DmlExecutionProvider": ("GPU", "DirectML - accelerates on Radeon 860M"),
            "VitisAIExecutionProvider": ("NPU", "AMD Ryzen AI - runs on XDNA 2 NPU"),
        }

        from tabulate import tabulate
        table = []
        for p in providers:
            info = provider_info.get(p, ("?", ""))
            table.append([p, info[0], info[1], "YES"])

        # Show missing interesting providers
        for p, info in provider_info.items():
            if p not in providers:
                table.append([p, info[0], info[1], "NOT INSTALLED"])

        print(tabulate(table, headers=["Provider", "Target", "Description", "Available"],
                       tablefmt="simple_grid"))
        return providers

    except ImportError:
        print("  ONNX Runtime not installed.")
        print("  Install with: pip install onnxruntime-directml")
        return []


# ---------------------------------------------------------------------------
# 4. NPU Inference Test (if VitisAI EP available)
# ---------------------------------------------------------------------------

def test_npu_inference(providers: list):
    """Attempt a simple inference on the NPU if VitisAI EP is available."""
    print("\n" + "=" * 70)
    print("  NPU INFERENCE TEST")
    print("=" * 70)

    if "VitisAIExecutionProvider" not in providers:
        print("""
  VitisAI Execution Provider is NOT installed.
  This is required to run inference on the AMD XDNA 2 NPU.

  To enable NPU acceleration:
  1. Install AMD Ryzen AI Software SDK 1.7.1 from:
     https://ryzenai.docs.amd.com/en/latest/inst.html

  2. Install the NPU driver (included in Ryzen AI Software)

  3. Install SDK wheels into a conda env (Python 3.12):
     pip install --no-deps <SDK_PATH>/onnxruntime_vitisai-*.whl
     pip install --no-deps <SDK_PATH>/voe-*.whl
     pip install numpy<2 protobuf flatbuffers coloredlogs packaging sympy

  4. Set environment variables:
     $env:XLNX_VART_FIRMWARE = "C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"

  Once installed, you can run ONNX models on the NPU like this:

     import onnxruntime as ort
     session = ort.InferenceSession(
         "model.onnx",
         providers=["VitisAIExecutionProvider"]
     )
     result = session.run(None, {"input": data})

  The NPU excels at:
    - INT8/INT4 quantized neural network inference
    - Transformer attention blocks
    - Convolutional neural networks
    - Sustained low-power AI workloads
""")
    else:
        print("  VitisAI EP detected! Running NPU inference test...")
        try:
            import onnxruntime as ort
            import numpy as np

            # Create a minimal ONNX model: Y = ReLU(X @ W)
            try:
                import onnx
                from onnx import helper, TensorProto
                import tempfile, os

                X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 64])
                Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 64])
                W = helper.make_tensor('W', TensorProto.FLOAT, [64, 64],
                    np.random.randn(64, 64).astype(np.float32).flatten().tolist())
                matmul = helper.make_node('MatMul', ['X', 'W'], ['matmul_out'])
                relu = helper.make_node('Relu', ['matmul_out'], ['Y'])
                graph = helper.make_graph([matmul, relu], 'npu_test', [X], [Y], [W])
                model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
                model.ir_version = 8

                tmp_path = os.path.join(tempfile.gettempdir(), 'npu_test.onnx')
                onnx.save(model, tmp_path)

                sess = ort.InferenceSession(tmp_path, providers=['VitisAIExecutionProvider'])
                inp = np.random.randn(1, 64).astype(np.float32)
                result = sess.run(None, {'X': inp})
                print(f"  Inference SUCCESS! Output shape: {result[0].shape}")
                print(f"  Active providers: {sess.get_providers()}")

                os.remove(tmp_path)
            except ImportError:
                print("  onnx package not installed - install with: pip install onnx")
                print("  VitisAI EP is available and ready for model inference.")
        except Exception as e:
            print(f"  Error testing NPU: {e}")


# ---------------------------------------------------------------------------
# 5. Architecture Overview
# ---------------------------------------------------------------------------

def print_architecture_overview():
    """Print the XDNA 2 NPU architecture overview."""
    print("\n" + "=" * 70)
    print("  AMD XDNA 2 NPU ARCHITECTURE OVERVIEW")
    print("=" * 70)
    print("""
  AMD Ryzen AI 7 PRO 350 - XDNA 2 NPU Architecture
  ===================================================

  ┌─────────────────────────────────────────────────┐
  │              AMD Ryzen AI 7 PRO 350             │
  │                                                 │
  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
  │  │ CPU      │  │ GPU      │  │ NPU          │  │
  │  │ Zen 5    │  │ RDNA 3.5 │  │ XDNA 2       │  │
  │  │ 8C/16T   │  │ 860M     │  │ ~50 TOPS     │  │
  │  │          │  │ ~4GB     │  │              │  │
  │  │ General  │  │ Graphics │  │ AI Inference │  │
  │  │ compute  │  │ + ML     │  │ INT4/8/FP16  │  │
  │  └──────────┘  └──────────┘  └──────────────┘  │
  │                     │                           │
  │              ┌──────┴──────┐                    │
  │              │ Shared DDR5 │                    │
  │              │   Memory    │                    │
  │              └─────────────┘                    │
  └─────────────────────────────────────────────────┘

  XDNA 2 NPU Details:
  ────────────────────
  • Array of AI Engine (AIE) tiles in a spatial architecture
  • Each tile has its own compute + local memory
  • Tiles interconnected via Network-on-Chip (NoC)
  • Hardware support for INT4, INT8, FP16, BF16
  • Dedicated DMA engines for data movement
  • Power-efficient: runs AI workloads at fraction of GPU power

  Programming Model:
  ──────────────────
  • Primary: ONNX Runtime + VitisAI Execution Provider
  • Models must be quantized (INT8/INT4) for best performance
  • Compiler maps ONNX graph → AIE tile instructions
  • Supports: CNNs, Transformers, LLMs (quantized)

  Typical Workflow:
  ─────────────────
  1. Train model (PyTorch/TensorFlow) on GPU/cloud
  2. Export to ONNX format
  3. Quantize model (INT8/INT4) using Olive or ONNX quantization
  4. Run inference via ONNX Runtime with VitisAI EP
  5. NPU compiler automatically maps ops to AIE tiles
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    npu_info = detect_npu_hardware()
    if npu_info:
        check_npu_driver()
    providers = check_onnxrt_providers()
    test_npu_inference(providers)
    print_architecture_overview()
