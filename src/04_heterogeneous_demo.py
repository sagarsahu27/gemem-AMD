"""
Heterogeneous Computing Demo: CPU vs GPU vs NPU
=================================================
Demonstrates running the same workload across all three compute units
in your AMD Ryzen AI 7 PRO 350 system:

  - CPU: NumPy / ONNX Runtime CPUExecutionProvider
  - GPU: OpenCL kernel / ONNX Runtime DmlExecutionProvider (DirectML)
  - NPU: ONNX Runtime VitisAIExecutionProvider (if available)

Run: conda run -n ryzen-ai-1.7.1 python src/04_heterogeneous_demo.py
"""

import time
import numpy as np
import os
import tempfile

# ---------------------------------------------------------------------------
# Helper: Create a simple ONNX model (MatMul + Relu)
# ---------------------------------------------------------------------------

def create_test_onnx_model(M=512, K=512, N=512) -> str:
    """Create a minimal ONNX model: Y = ReLU(X @ W) for benchmarking."""
    try:
        import onnxruntime  # noqa: F401 - just checking availability
    except ImportError:
        raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime-directml")

    # Build ONNX model manually using numpy protobuf
    # We'll use onnx library if available, otherwise create manually
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper

        W_data = np.random.randn(K, N).astype(np.float32) * 0.01

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
        W = numpy_helper.from_array(W_data, name="W")

        matmul_node = helper.make_node("MatMul", ["X", "W"], ["matmul_out"])
        relu_node = helper.make_node("Relu", ["matmul_out"], ["Y"])

        graph = helper.make_graph([matmul_node, relu_node], "test_matmul_relu",
                                  [X], [Y], [W])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8

        path = os.path.join(tempfile.gettempdir(), "amd_hetero_test.onnx")
        onnx.save(model, path)
        return path

    except ImportError:
        # Fallback: build ONNX protobuf manually (no onnx package needed)
        return _create_onnx_manual(M, K, N)


def _create_onnx_manual(M, K, N) -> str:
    """Create ONNX model without the onnx package using raw protobuf-like bytes.
    Uses a simpler approach: just test with onnxruntime directly."""
    # If we can't create an ONNX file, we'll skip the ONNX-based test
    return None


# ---------------------------------------------------------------------------
# Benchmark: CPU (NumPy)
# ---------------------------------------------------------------------------

def benchmark_cpu_numpy(M=512, K=512, N=512, iterations=20):
    """Matrix multiply benchmark on CPU using NumPy."""
    print("\n  [CPU - NumPy]")
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)

    # Warmup
    for _ in range(3):
        _ = np.maximum(X @ W, 0)

    # Timed
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        Y = np.maximum(X @ W, 0)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    gflops = (2 * M * K * N) / (np.mean(times) * 1e9)
    print(f"    Average: {avg_ms:.2f} ms (+/- {std_ms:.2f} ms)")
    print(f"    Throughput: {gflops:.1f} GFLOPS")
    return avg_ms


# ---------------------------------------------------------------------------
# Benchmark: GPU (OpenCL)
# ---------------------------------------------------------------------------

def benchmark_gpu_opencl(M=512, K=512, N=512, iterations=20):
    """Matrix multiply benchmark on GPU using OpenCL."""
    print("\n  [GPU - OpenCL (Radeon 860M)]")

    try:
        import pyopencl as cl
    except ImportError:
        print("    pyopencl not available, skipping.")
        return None

    # Find AMD GPU
    gpu_device = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                gpu_device = device
                break
        if gpu_device:
            break

    if not gpu_device:
        print("    No GPU device found.")
        return None

    print(f"    Device: {gpu_device.name}")

    ctx = cl.Context([gpu_device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    kernel_src = """
    __kernel void matmul_relu(__global const float* X,
                              __global const float* W,
                              __global float* Y,
                              const int M, const int K, const int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += X[row * K + k] * W[k * N + col];
            }
            Y[row * N + col] = max(sum, 0.0f);  // ReLU
        }
    }
    """

    prg = cl.Program(ctx, kernel_src).build()
    mf = cl.mem_flags

    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)

    x_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    w_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=W)
    y_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)

    Y = np.empty((M, N), dtype=np.float32)

    # Warmup
    for _ in range(3):
        evt = prg.matmul_relu(queue, (M, N), None, x_buf, w_buf, y_buf,
                               np.int32(M), np.int32(K), np.int32(N))
        evt.wait()

    # Timed
    times = []
    kernel_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        evt = prg.matmul_relu(queue, (M, N), None, x_buf, w_buf, y_buf,
                               np.int32(M), np.int32(K), np.int32(N))
        evt.wait()
        cl.enqueue_copy(queue, Y, y_buf).wait()
        times.append(time.perf_counter() - t0)
        kernel_times.append((evt.profile.end - evt.profile.start) / 1e6)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    kernel_avg = np.mean(kernel_times)
    gflops = (2 * M * K * N) / (np.mean(times) * 1e9)
    print(f"    Average (total): {avg_ms:.2f} ms (+/- {std_ms:.2f} ms)")
    print(f"    Average (kernel): {kernel_avg:.2f} ms")
    print(f"    Throughput: {gflops:.1f} GFLOPS")
    return avg_ms


# ---------------------------------------------------------------------------
# Benchmark: GPU (DirectML via ONNX Runtime)
# ---------------------------------------------------------------------------

def benchmark_gpu_directml(model_path: str, M=512, K=512, iterations=20):
    """Benchmark ONNX model inference on GPU via DirectML."""
    print("\n  [GPU - DirectML (Radeon 860M)]")

    if not model_path:
        print("    No ONNX model available, skipping.")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("    onnxruntime not available, skipping.")
        return None

    if "DmlExecutionProvider" not in ort.get_available_providers():
        print("    DirectML EP not available, skipping.")
        return None

    session = ort.InferenceSession(model_path, providers=["DmlExecutionProvider"])
    X = np.random.randn(M, K).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {"X": X})

    # Timed
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = session.run(None, {"X": X})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"    Average: {avg_ms:.2f} ms (+/- {std_ms:.2f} ms)")
    return avg_ms


# ---------------------------------------------------------------------------
# Benchmark: CPU (ONNX Runtime)
# ---------------------------------------------------------------------------

def benchmark_cpu_onnxrt(model_path: str, M=512, K=512, iterations=20):
    """Benchmark ONNX model inference on CPU."""
    print("\n  [CPU - ONNX Runtime]")

    if not model_path:
        print("    No ONNX model available, skipping.")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("    onnxruntime not available, skipping.")
        return None

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    X = np.random.randn(M, K).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {"X": X})

    # Timed
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = session.run(None, {"X": X})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"    Average: {avg_ms:.2f} ms (+/- {std_ms:.2f} ms)")
    return avg_ms


# ---------------------------------------------------------------------------
# Benchmark: NPU (ONNX Runtime + VitisAI EP)
# ---------------------------------------------------------------------------

def benchmark_npu(model_path: str, M=512, K=512, iterations=20):
    """Benchmark ONNX model inference on NPU via VitisAI EP."""
    print("\n  [NPU - XDNA 2 (VitisAI EP)]")

    if not model_path:
        print("    No ONNX model available, skipping.")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print("    onnxruntime not available, skipping.")
        return None

    if "VitisAIExecutionProvider" not in ort.get_available_providers():
        print("    VitisAI EP not installed.")
        print("    Install AMD Ryzen AI Software to enable NPU inference.")
        print("    See: https://ryzenai.docs.amd.com/en/latest/inst.html")
        return None

    # Set xclbin firmware path if not already set
    if not os.environ.get("XLNX_VART_FIRMWARE"):
        xclbin = r"C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
        if os.path.exists(xclbin):
            os.environ["XLNX_VART_FIRMWARE"] = xclbin

    session = ort.InferenceSession(model_path, providers=["VitisAIExecutionProvider"])
    X = np.random.randn(M, K).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {"X": X})

    # Timed
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = session.run(None, {"X": X})
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"    Average: {avg_ms:.2f} ms (+/- {std_ms:.2f} ms)")
    return avg_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  HETEROGENEOUS COMPUTING DEMO")
    print("  AMD Ryzen AI 7 PRO 350: CPU vs GPU vs NPU")
    print("=" * 70)

    M, K, N = 512, 512, 512
    iterations = 20

    print(f"\n  Workload: MatMul({M}x{K} @ {K}x{N}) + ReLU")
    print(f"  Iterations: {iterations}")

    # --- Raw compute benchmarks ---
    print("\n" + "-" * 70)
    print("  PART 1: Raw Compute (NumPy vs OpenCL)")
    print("-" * 70)

    cpu_time = benchmark_cpu_numpy(M, K, N, iterations)
    gpu_cl_time = benchmark_gpu_opencl(M, K, N, iterations)

    # --- ONNX Runtime benchmarks ---
    print("\n" + "-" * 70)
    print("  PART 2: ONNX Runtime Inference (CPU vs GPU vs NPU)")
    print("-" * 70)

    model_path = None
    try:
        model_path = create_test_onnx_model(M, K, N)
    except Exception as e:
        print(f"\n  Could not create ONNX model: {e}")
        print("  Install 'onnx' package for ONNX benchmarks: pip install onnx")

    cpu_ort_time = benchmark_cpu_onnxrt(model_path, M, K, iterations)
    gpu_dml_time = benchmark_gpu_directml(model_path, M, K, iterations)
    npu_time = benchmark_npu(model_path, M, K, iterations)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    from tabulate import tabulate
    results = []
    if cpu_time:
        results.append(["CPU (NumPy)", f"{cpu_time:.2f} ms", "1.00x"])
    if gpu_cl_time and cpu_time:
        results.append(["GPU (OpenCL)", f"{gpu_cl_time:.2f} ms", f"{cpu_time/gpu_cl_time:.2f}x"])
    if cpu_ort_time:
        results.append(["CPU (ONNX RT)", f"{cpu_ort_time:.2f} ms",
                        f"{cpu_time/cpu_ort_time:.2f}x" if cpu_time else "N/A"])
    if gpu_dml_time and cpu_time:
        results.append(["GPU (DirectML)", f"{gpu_dml_time:.2f} ms", f"{cpu_time/gpu_dml_time:.2f}x"])
    if npu_time and cpu_time:
        results.append(["NPU (VitisAI)", f"{npu_time:.2f} ms", f"{cpu_time/npu_time:.2f}x"])

    if results:
        print(tabulate(results, headers=["Compute Unit", "Avg Time", "Speedup vs CPU"],
                       tablefmt="simple_grid"))

    # Clean up
    if model_path and os.path.exists(model_path):
        os.remove(model_path)

    print("""
  KEY TAKEAWAYS:
  ──────────────
  • CPU (Zen 5): Best for general-purpose, low-latency, and serial workloads
  • GPU (RDNA 3.5): Best for parallel compute, graphics, and ML inference
  • NPU (XDNA 2): Best for sustained, power-efficient AI inference (INT8/INT4)

  The ideal heterogeneous strategy assigns each workload to the best unit:
    - Data preprocessing → CPU
    - Training & large batch inference → GPU (DirectML)
    - Always-on AI features → NPU (low power, high efficiency)
""")


if __name__ == "__main__":
    main()
