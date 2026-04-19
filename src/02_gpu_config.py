"""
AMD GPU Configuration Deep-Dive
================================
Explores your AMD Radeon 860M (RDNA 3.5) integrated GPU capabilities:
  1. OpenCL device properties (compute units, memory, clock speeds)
  2. DirectML availability and device enumeration
  3. Simple GPU compute benchmark (matrix multiply via OpenCL)

Run: uv run python src/02_gpu_config.py
"""

import sys
import time
import numpy as np


# ---------------------------------------------------------------------------
# 1. OpenCL GPU Exploration
# ---------------------------------------------------------------------------

def explore_opencl():
    """Enumerate OpenCL platforms and devices, print GPU properties."""
    print("=" * 70)
    print("  OPENCL GPU EXPLORATION")
    print("=" * 70)

    try:
        import pyopencl as cl
    except ImportError:
        print("  pyopencl not installed. Install with: uv add pyopencl")
        return False

    platforms = cl.get_platforms()
    if not platforms:
        print("  No OpenCL platforms found.")
        return False

    from tabulate import tabulate

    for plat_idx, platform in enumerate(platforms):
        print(f"\n  Platform {plat_idx}: {platform.name}")
        print(f"  Vendor:  {platform.vendor}")
        print(f"  Version: {platform.version}")

        devices = platform.get_devices()
        for dev_idx, device in enumerate(devices):
            is_gpu = device.type == cl.device_type.GPU
            dev_type = "GPU" if is_gpu else ("CPU" if device.type == cl.device_type.CPU else "OTHER")

            props = [
                ["Device Type", dev_type],
                ["Name", device.name],
                ["Vendor", device.vendor],
                ["Driver Version", device.driver_version],
                ["OpenCL Version", device.version],
                ["Compute Units", device.max_compute_units],
                ["Max Clock (MHz)", device.max_clock_frequency],
                ["Global Memory (MB)", round(device.global_mem_size / (1024**2))],
                ["Local Memory (KB)", round(device.local_mem_size / 1024)],
                ["Max Work Group Size", device.max_work_group_size],
                ["Max Work Item Dims", device.max_work_item_dimensions],
                ["Max Work Item Sizes", list(device.max_work_item_sizes)],
                ["Image Support", bool(device.image_support)],
                ["Double FP Support", bool(device.double_fp_config)],
                ["Half FP Support", bool(device.half_fp_config)],
                ["Preferred Vector Width (float)", device.preferred_vector_width_float],
                ["Preferred Vector Width (half)", device.preferred_vector_width_half],
            ]

            # Try to get extra AMD-specific properties
            try:
                props.append(["Max Memory Alloc (MB)", round(device.max_mem_alloc_size / (1024**2))])
            except Exception:
                pass

            print(f"\n  Device {dev_idx} [{dev_type}]:")
            print(tabulate(props, tablefmt="simple_grid"))

    return True


# ---------------------------------------------------------------------------
# 2. DirectML (via ONNX Runtime) GPU Check
# ---------------------------------------------------------------------------

def explore_directml():
    """Check DirectML availability via onnxruntime-directml."""
    print("\n" + "=" * 70)
    print("  DIRECTML GPU EXPLORATION")
    print("=" * 70)

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime-directml not installed.")
        return False

    providers = ort.get_available_providers()
    print(f"\n  Available Execution Providers:")
    for p in providers:
        tag = " <-- GPU accelerated" if "Dml" in p else ""
        print(f"    - {p}{tag}")

    dml_ok = "DmlExecutionProvider" in providers
    if dml_ok:
        print("\n  DirectML Status: READY")
        print("  Your AMD Radeon 860M can accelerate ONNX inference via DirectML.")
        print("  DirectML supports: convolutions, matrix ops, attention, and 1400+ ONNX ops.")
    else:
        print("\n  DirectML Status: NOT AVAILABLE")
        print("  Install with: uv add onnxruntime-directml")

    return dml_ok


# ---------------------------------------------------------------------------
# 3. GPU Compute Benchmark - OpenCL Matrix Multiply
# ---------------------------------------------------------------------------

def gpu_benchmark():
    """Run a simple matrix multiply on the GPU via OpenCL vs CPU via NumPy."""
    print("\n" + "=" * 70)
    print("  GPU COMPUTE BENCHMARK (Matrix Multiply)")
    print("=" * 70)

    try:
        import pyopencl as cl
    except ImportError:
        print("  Skipping benchmark (pyopencl not available).")
        return

    # Find an AMD GPU device
    gpu_device = None
    gpu_platform = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                gpu_device = device
                gpu_platform = platform
                break
        if gpu_device:
            break

    if not gpu_device:
        print("  No GPU device found for benchmark.")
        return

    print(f"  Using: {gpu_device.name}")

    # Matrix size
    N = 1024
    print(f"  Matrix size: {N}x{N} (float32)\n")

    # Create random matrices
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # --- CPU baseline (NumPy) ---
    t0 = time.perf_counter()
    C_cpu = A @ B
    cpu_time = time.perf_counter() - t0

    # --- GPU (OpenCL) ---
    ctx = cl.Context([gpu_device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # OpenCL kernel for naive matrix multiply
    kernel_src = """
    __kernel void matmul(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    """

    prg = cl.Program(ctx, kernel_src).build()
    mf = cl.mem_flags

    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C_cpu.nbytes)

    C_gpu = np.empty_like(C_cpu)

    # Warm-up run
    evt = prg.matmul(queue, (N, N), None, a_buf, b_buf, c_buf, np.int32(N))
    evt.wait()

    # Timed run
    t0 = time.perf_counter()
    evt = prg.matmul(queue, (N, N), None, a_buf, b_buf, c_buf, np.int32(N))
    evt.wait()
    cl.enqueue_copy(queue, C_gpu, c_buf).wait()
    gpu_time = time.perf_counter() - t0

    # Kernel-only time from profiling
    kernel_ns = evt.profile.end - evt.profile.start
    kernel_ms = kernel_ns / 1e6

    # Verify correctness
    max_error = np.max(np.abs(C_cpu - C_gpu))
    gflops = (2 * N**3) / 1e9

    from tabulate import tabulate
    results = [
        ["CPU (NumPy)", f"{cpu_time*1000:.1f} ms", f"{gflops/cpu_time:.1f} GFLOPS"],
        ["GPU (OpenCL)", f"{gpu_time*1000:.1f} ms (total)", f"{gflops/gpu_time:.1f} GFLOPS"],
        ["GPU kernel only", f"{kernel_ms:.1f} ms", f"{gflops/(kernel_ms/1000):.1f} GFLOPS"],
    ]
    print(tabulate(results, headers=["Target", "Time", "Throughput"], tablefmt="simple_grid"))
    print(f"\n  Max error (CPU vs GPU): {max_error:.2e}")
    print(f"  Result: {'PASS' if max_error < 0.1 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    opencl_ok = explore_opencl()
    directml_ok = explore_directml()

    if opencl_ok:
        gpu_benchmark()

    print("\n" + "=" * 70)
    print("  GPU PROGRAMMING NOTES FOR AMD RADEON 860M")
    print("=" * 70)
    print("""
  Your Radeon 860M (RDNA 3.5) supports:
    - OpenCL 2.0+ for general GPU compute
    - DirectML for accelerated ML inference on Windows
    - Vulkan Compute shaders (via Vulkan SDK)

  Key specs (RDNA 3.5 iGPU):
    - Shared system memory (up to ~4GB dedicated equivalent)
    - Hardware ray tracing (RT) support
    - FP16/BF16 acceleration for AI workloads
    - Wave32/Wave64 execution modes

  Next steps:
    - Run 04_heterogeneous_demo.py to see CPU vs GPU vs NPU inference
    - Try writing custom OpenCL kernels in 02_gpu_config.py
    - Explore DirectML with ONNX models
""")
