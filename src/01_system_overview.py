"""
AMD Heterogeneous Programming - System Overview
================================================
Detects and reports all compute units in your AMD Ryzen AI system:
  - CPU (Zen 5 cores)
  - GPU (RDNA 3.5 integrated graphics)
  - NPU (XDNA 2 AI accelerator)

Run: uv run python src/01_system_overview.py
"""

import platform
import os
import subprocess
import ctypes
from collections import defaultdict

# ---------------------------------------------------------------------------
# CPU Detection
# ---------------------------------------------------------------------------

def get_cpu_info() -> dict:
    """Gather CPU details using WMI."""
    info = {
        "name": "Unknown",
        "cores_physical": os.cpu_count(),
        "cores_logical": os.cpu_count(),
        "architecture": platform.machine(),
        "features": [],
    }

    try:
        import wmi
        c = wmi.WMI()
        for proc in c.Win32_Processor():
            info["name"] = proc.Name.strip()
            info["cores_physical"] = proc.NumberOfCores
            info["cores_logical"] = proc.NumberOfLogicalProcessors
            info["max_clock_mhz"] = proc.MaxClockSpeed
            info["l2_cache_kb"] = proc.L2CacheSize
            info["l3_cache_kb"] = proc.L3CacheSize
            break
    except ImportError:
        # Fallback: read from environment / platform
        info["name"] = platform.processor() or "Unknown"

    return info


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------

def get_gpu_info() -> list[dict]:
    """Detect AMD GPUs via WMI and DirectML."""
    gpus = []

    # --- WMI approach ---
    try:
        import wmi
        c = wmi.WMI()
        for gpu in c.Win32_VideoController():
            if gpu.Name and "DisplayLink" not in gpu.Name:
                raw_ram = int(gpu.AdapterRAM) if gpu.AdapterRAM else 0
                # AdapterRAM is unsigned 32-bit but WMI returns signed; handle overflow
                if raw_ram < 0:
                    raw_ram += 2**32
                gpus.append({
                    "name": gpu.Name.strip(),
                    "driver_version": gpu.DriverVersion,
                    "vram_bytes": raw_ram,
                    "vram_gb": round(raw_ram / (1024**3), 1),
                    "status": gpu.Status,
                    "video_processor": gpu.VideoProcessor,
                })
    except ImportError:
        pass

    # --- DirectML device enumeration ---
    try:
        import onnxruntime as ort
        dml_available = "DmlExecutionProvider" in ort.get_available_providers()
        if dml_available and gpus:
            gpus[0]["directml_supported"] = True
    except Exception:
        pass

    return gpus


# ---------------------------------------------------------------------------
# NPU Detection
# ---------------------------------------------------------------------------

def get_npu_info() -> dict | None:
    """Detect the AMD XDNA NPU via PnP device enumeration."""
    npu = None

    try:
        import wmi
        c = wmi.WMI()
        for device in c.Win32_PnPEntity():
            name = device.Name or ""
            if "NPU" in name.upper() and "Compute Accelerator" in name:
                npu = {
                    "name": name.strip(),
                    "device_id": device.DeviceID,
                    "status": device.Status,
                    "manufacturer": device.Manufacturer,
                }
                break
    except ImportError:
        # Fallback: use PowerShell
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-PnpDevice | Where-Object { $_.FriendlyName -match 'NPU Compute' } "
                 "| Select-Object FriendlyName, Status, InstanceId | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10
            )
            import json
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                data = [data]
            if data:
                d = data[0]
                npu = {
                    "name": d.get("FriendlyName", "Unknown").strip(),
                    "device_id": d.get("InstanceId", ""),
                    "status": d.get("Status", "Unknown"),
                }
        except Exception:
            pass

    # Check for ONNX Runtime NPU execution provider
    if npu:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            npu["onnxrt_providers"] = providers
            npu["vitisai_ep_available"] = "VitisAIExecutionProvider" in providers
        except Exception:
            npu["vitisai_ep_available"] = False

    return npu


# ---------------------------------------------------------------------------
# ONNX Runtime Provider Inventory
# ---------------------------------------------------------------------------

def get_onnxrt_providers() -> list[str]:
    """List all ONNX Runtime execution providers available."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pretty Print
# ---------------------------------------------------------------------------

def print_report():
    from tabulate import tabulate

    print("=" * 70)
    print("  AMD HETEROGENEOUS SYSTEM OVERVIEW")
    print("=" * 70)

    # ---- CPU ----
    cpu = get_cpu_info()
    print("\n[CPU - Zen 5]")
    cpu_table = [
        ["Processor", cpu["name"]],
        ["Physical Cores", cpu["cores_physical"]],
        ["Logical Processors", cpu["cores_logical"]],
        ["Architecture", cpu["architecture"]],
    ]
    if "max_clock_mhz" in cpu:
        cpu_table.append(["Max Clock (MHz)", cpu["max_clock_mhz"]])
    if "l2_cache_kb" in cpu:
        cpu_table.append(["L2 Cache (KB)", cpu["l2_cache_kb"]])
    if "l3_cache_kb" in cpu:
        cpu_table.append(["L3 Cache (KB)", cpu["l3_cache_kb"]])
    print(tabulate(cpu_table, tablefmt="simple_grid"))

    # ---- GPU ----
    gpus = get_gpu_info()
    print("\n[GPU - RDNA 3.5]")
    if gpus:
        for i, gpu in enumerate(gpus):
            gpu_table = [
                ["GPU #", i],
                ["Name", gpu["name"]],
                ["Driver", gpu["driver_version"]],
                ["VRAM", f"{gpu['vram_gb']} GB"],
                ["Status", gpu["status"]],
            ]
            if gpu.get("directml_supported"):
                gpu_table.append(["DirectML", "Supported"])
            print(tabulate(gpu_table, tablefmt="simple_grid"))
    else:
        print("  No AMD GPU detected.")

    # ---- NPU ----
    npu = get_npu_info()
    print("\n[NPU - XDNA 2]")
    if npu:
        npu_table = [
            ["Name", npu["name"]],
            ["Device ID", npu["device_id"][:60] + "..."],
            ["Status", npu["status"]],
            ["VitisAI EP", "Yes" if npu.get("vitisai_ep_available") else "Not installed"],
        ]
        print(tabulate(npu_table, tablefmt="simple_grid"))
    else:
        print("  No AMD NPU detected.")

    # ---- ONNX Runtime ----
    providers = get_onnxrt_providers()
    print("\n[ONNX Runtime Execution Providers]")
    if providers:
        for p in providers:
            marker = " <-- GPU" if "Dml" in p else (" <-- NPU" if "Vitis" in p else "")
            print(f"  - {p}{marker}")
    else:
        print("  ONNX Runtime not available.")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  HETEROGENEOUS COMPUTE SUMMARY")
    print("=" * 70)
    summary = [
        ["CPU", cpu["name"], "Ready"],
        ["GPU", gpus[0]["name"] if gpus else "N/A",
         "DirectML ready" if gpus and gpus[0].get("directml_supported") else ("Driver OK" if gpus else "N/A")],
        ["NPU", npu["name"] if npu else "N/A",
         npu["status"] if npu else "Not detected"],
    ]
    print(tabulate(summary, headers=["Unit", "Device", "Status"], tablefmt="simple_grid"))
    print()


if __name__ == "__main__":
    print_report()
