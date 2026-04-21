"""
Session 4: Heterogeneous Orchestration — CPU + GPU + NPU Pipeline
=================================================================
Combine all three compute units into a real pipeline.

Topics:
  1. Pipeline pattern: CPU preprocess → GPU compute → NPU inference
  2. Task-parallel decomposition across devices
  3. Double-buffering and overlapping compute with data transfer
  4. Choosing the right device for each stage
  5. Real-world architecture patterns

Hardware: CPU (Zen 5) + GPU (RDNA 3.5) + NPU (XDNA 2)
Run: conda run -n ryzen-ai-1.7.1 python tutorials/session4_heterogeneous.py
"""

import numpy as np
import pyopencl as cl
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper
import os
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

TEMP_DIR = tempfile.gettempdir()


# ===================================================================
# SETUP
# ===================================================================

def get_gpu_context():
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                return ctx, queue, device
    raise RuntimeError("No GPU found")


def create_classifier_model(input_dim, hidden_dim, num_classes):
    """Create a simple MLP classifier ONNX model."""
    W1 = numpy_helper.from_array(
        np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01, name="W1")
    B1 = numpy_helper.from_array(np.zeros(hidden_dim, dtype=np.float32), name="B1")
    W2 = numpy_helper.from_array(
        np.random.randn(hidden_dim, num_classes).astype(np.float32) * 0.01, name="W2")
    B2 = numpy_helper.from_array(np.zeros(num_classes, dtype=np.float32), name="B2")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, num_classes])

    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["mm1"]),
        helper.make_node("Add", ["mm1", "B1"], ["h1"]),
        helper.make_node("Relu", ["h1"], ["r1"]),
        helper.make_node("MatMul", ["r1", "W2"], ["mm2"]),
        helper.make_node("Add", ["mm2", "B2"], ["Y"]),
    ]

    graph = helper.make_graph(nodes, "classifier", [X], [Y], [W1, B1, W2, B2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


# ===================================================================
# PATTERN 10: PIPELINE — CPU → GPU → NPU
# ===================================================================
# Real-world AI systems use pipelines:
#   Stage 1 (CPU): Data loading, decoding, preprocessing
#   Stage 2 (GPU): Feature extraction, heavy compute
#   Stage 3 (NPU): Classification / inference
#
# Interview Q: "Design a real-time AI pipeline for edge devices"
# Answer: Assign each stage to the best device, overlap execution.
# ===================================================================

GPU_NORMALIZE_KERNEL = """
// Normalize features: (x - mean) / std
__kernel void normalize(__global const float* input,
                        __global float* output,
                        const float mean,
                        const float inv_std,
                        const int N) {
    int gid = get_global_id(0);
    if (gid < N) {
        output[gid] = (input[gid] - mean) * inv_std;
    }
}
"""

GPU_FEATURE_KERNEL = """
// Simple feature extraction: sliding window statistics
__kernel void extract_features(__global const float* input,
                               __global float* features,
                               const int input_len,
                               const int window_size,
                               const int num_features) {
    int fid = get_global_id(0);  // Feature index
    if (fid >= num_features) return;

    int start = fid * window_size;
    int end = min(start + window_size, input_len);

    // Compute mean and variance for this window
    float sum = 0.0f, sq_sum = 0.0f;
    int count = 0;
    for (int i = start; i < end; i++) {
        float val = input[i];
        sum += val;
        sq_sum += val * val;
        count++;
    }

    float mean = sum / count;
    float var = sq_sum / count - mean * mean;

    // Output: [mean, variance] for each window
    features[fid * 2] = mean;
    features[fid * 2 + 1] = var;
}
"""

def demo_pipeline(ctx, queue):
    """Demonstrate a CPU → GPU → NPU inference pipeline."""
    print("\n" + "=" * 70)
    print("  PATTERN 10: CPU → GPU → NPU PIPELINE")
    print("  Preprocess (CPU) → Features (GPU) → Classify (NPU)")
    print("=" * 70)

    # Configuration
    RAW_LEN = 4096    # Raw input signal length
    WINDOW = 32       # Feature extraction window
    NUM_FEATURES = RAW_LEN // WINDOW  # 128 windows
    FEATURE_DIM = NUM_FEATURES * 2     # mean + variance per window = 256
    NUM_CLASSES = 10

    # Build NPU classifier model
    model = create_classifier_model(FEATURE_DIM, 128, NUM_CLASSES)
    model_path = os.path.join(TEMP_DIR, "pipeline_classifier.onnx")
    onnx.save(model, model_path)

    # NPU session
    npu_available = "VitisAIExecutionProvider" in ort.get_available_providers()
    ep = "VitisAIExecutionProvider" if npu_available else "CPUExecutionProvider"
    npu_sess = ort.InferenceSession(model_path, providers=[ep])
    device_label = "NPU" if npu_available else "CPU (NPU unavailable)"

    # GPU kernels
    prg = cl.Program(ctx, GPU_NORMALIZE_KERNEL + GPU_FEATURE_KERNEL).build()
    k_norm = cl.Kernel(prg, "normalize")
    k_feat = cl.Kernel(prg, "extract_features")

    mf = cl.mem_flags
    NUM_SAMPLES = 100
    total_times = {"cpu_preprocess": [], "gpu_compute": [], "npu_inference": [], "total": []}

    print(f"\n  Pipeline: {RAW_LEN} raw samples → {FEATURE_DIM} features → {NUM_CLASSES} classes")
    print(f"  Classifier on: {device_label}")
    print(f"  Processing {NUM_SAMPLES} samples...\n")

    for i in range(NUM_SAMPLES):
        t_total = time.perf_counter()

        # ── Stage 1: CPU Preprocessing ──
        t0 = time.perf_counter()
        raw = np.random.randn(RAW_LEN).astype(np.float32) * 100  # Simulate sensor data
        # Simple preprocessing: clip outliers, scale
        raw = np.clip(raw, -300, 300)
        mean_val = float(np.mean(raw))
        std_val = float(np.std(raw))
        total_times["cpu_preprocess"].append(time.perf_counter() - t0)

        # ── Stage 2: GPU Feature Extraction ──
        t0 = time.perf_counter()
        d_raw = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=raw)
        d_norm = cl.Buffer(ctx, mf.READ_WRITE, raw.nbytes)
        d_feat = cl.Buffer(ctx, mf.WRITE_ONLY, NUM_FEATURES * 2 * 4)

        # Normalize on GPU
        inv_std = 1.0 / max(std_val, 1e-7)
        k_norm.set_args(d_raw, d_norm, np.float32(mean_val), np.float32(inv_std), np.int32(RAW_LEN))
        cl.enqueue_nd_range_kernel(queue, k_norm, (RAW_LEN,), None).wait()

        # Extract features on GPU
        k_feat.set_args(d_norm, d_feat, np.int32(RAW_LEN), np.int32(WINDOW), np.int32(NUM_FEATURES))
        cl.enqueue_nd_range_kernel(queue, k_feat, (NUM_FEATURES,), None).wait()

        features = np.empty(NUM_FEATURES * 2, dtype=np.float32)
        cl.enqueue_copy(queue, features, d_feat).wait()
        total_times["gpu_compute"].append(time.perf_counter() - t0)

        # ── Stage 3: NPU Classification ──
        t0 = time.perf_counter()
        features_2d = features.reshape(1, -1)
        logits = npu_sess.run(None, {"X": features_2d})[0]
        predicted_class = int(np.argmax(logits))
        total_times["npu_inference"].append(time.perf_counter() - t0)

        total_times["total"].append(time.perf_counter() - t_total)

    # Results
    print("  Pipeline Timing (average over {} samples):".format(NUM_SAMPLES))
    for stage, times in total_times.items():
        avg = np.mean(times) * 1000
        print(f"    {stage:20s}: {avg:7.3f} ms")

    total_avg = np.mean(total_times["total"]) * 1000
    throughput = 1000 / total_avg
    print(f"\n  End-to-end throughput: {throughput:.0f} samples/sec")

    os.remove(model_path)

    print("""
  KEY CONCEPTS:
  ─────────────
  • Each device does what it's best at:
    - CPU: data loading, preprocessing, control flow
    - GPU: parallel feature extraction, normalization
    - NPU: power-efficient neural network inference
  • Pipeline can be overlapped with double-buffering
  • Interview tip: Draw the pipeline diagram with data flow
""")


# ===================================================================
# PATTERN 11: TASK-PARALLEL DECOMPOSITION
# ===================================================================
# Run independent tasks on different devices simultaneously.
#
# Interview Q: "You have 3 independent models. How to maximize throughput?"
# Answer: Run each on a different device concurrently (CPU, GPU, NPU).
# ===================================================================

def demo_task_parallel():
    """Run independent tasks on CPU, GPU, NPU concurrently."""
    print("\n" + "=" * 70)
    print("  PATTERN 11: TASK-PARALLEL DECOMPOSITION")
    print("  Independent tasks on different devices simultaneously")
    print("=" * 70)

    # Create three independent models
    models = {}
    for name, dims in [("model_a", (128, 256, 64)),
                       ("model_b", (64, 128, 32)),
                       ("model_c", (256, 512, 128))]:
        m = create_classifier_model(*dims)
        path = os.path.join(TEMP_DIR, f"{name}.onnx")
        onnx.save(m, path)
        models[name] = (path, dims)

    available = ort.get_available_providers()

    # Map tasks to devices
    task_assignments = [
        ("model_a", "CPUExecutionProvider", "CPU"),
        ("model_b", "DmlExecutionProvider", "GPU"),
        ("model_c", "VitisAIExecutionProvider", "NPU"),
    ]

    # Filter to available providers
    task_assignments = [(n, ep, l) for n, ep, l in task_assignments if ep in available]

    def run_task(name, ep, label, iterations=50):
        path, dims = models[name]
        sess = ort.InferenceSession(path, providers=[ep])
        X = np.random.randn(1, dims[0]).astype(np.float32)
        # Warmup
        for _ in range(5):
            sess.run(None, {"X": X})
        # Timed
        t0 = time.perf_counter()
        for _ in range(iterations):
            sess.run(None, {"X": X})
        elapsed = time.perf_counter() - t0
        return label, name, elapsed * 1000 / iterations

    # Sequential execution
    print("\n  Sequential execution:")
    t0 = time.perf_counter()
    seq_results = []
    for name, ep, label in task_assignments:
        result = run_task(name, ep, label)
        seq_results.append(result)
        print(f"    {result[0]:5s} running {result[1]}: {result[2]:.3f} ms/iter")
    seq_total = (time.perf_counter() - t0) * 1000
    print(f"    Total: {seq_total:.1f} ms")

    # Parallel execution
    print("\n  Parallel execution (ThreadPoolExecutor):")
    t0 = time.perf_counter()
    par_results = []
    with ThreadPoolExecutor(max_workers=len(task_assignments)) as executor:
        futures = {executor.submit(run_task, n, ep, l): l
                   for n, ep, l in task_assignments}
        for future in as_completed(futures):
            result = future.result()
            par_results.append(result)
            print(f"    {result[0]:5s} running {result[1]}: {result[2]:.3f} ms/iter")
    par_total = (time.perf_counter() - t0) * 1000
    print(f"    Total: {par_total:.1f} ms")

    speedup = seq_total / par_total if par_total > 0 else 0
    print(f"\n  Parallel speedup: {speedup:.2f}x")

    for name, (path, _) in models.items():
        os.remove(path)

    print("""
  KEY CONCEPTS:
  ─────────────
  • Different devices can execute truly in parallel
  • CPU, GPU, NPU have independent execution engines
  • Use threads to submit work to each device concurrently
  • Speedup ≈ max(task_times) instead of sum(task_times)
  • Interview tip: This is Amdahl's law applied to heterogeneous systems
    - Speedup limited by the slowest device
    - Balance workloads across devices for maximum utilization
""")


# ===================================================================
# PATTERN 12: DEVICE SELECTION STRATEGY
# ===================================================================
# Interview Q: "How do you decide which device to use?"
# Answer: Decision tree based on workload characteristics.
# ===================================================================

def demo_device_selection():
    """Demonstrate device selection decision-making."""
    print("\n" + "=" * 70)
    print("  PATTERN 12: DEVICE SELECTION STRATEGY")
    print("  Matching workloads to the right compute unit")
    print("=" * 70)

    print("""
  Decision Tree: Which Device?
  ═══════════════════════════════════════════════════════════

  Is it a neural network inference?
  ├── YES: Is the model quantized (INT8/INT4)?
  │   ├── YES → NPU (XDNA 2)   [Best: power-efficient, dedicated]
  │   └── NO  → Is batch size > 16?
  │       ├── YES → GPU (DirectML)  [Best: parallel throughput]
  │       └── NO  → CPU or NPU     [Low overhead wins]
  │
  └── NO: Is it data-parallel (same op on many elements)?
      ├── YES: Is data size > 100K elements?
      │   ├── YES → GPU (OpenCL)   [Enough work to hide latency]
      │   └── NO  → CPU (NumPy)    [GPU launch overhead dominates]
      │
      └── NO: Is it control-flow heavy / sequential?
          └── YES → CPU (Zen 5)    [Best for branching, low latency]

  ═══════════════════════════════════════════════════════════
""")

    # Demonstrate with concrete measurements
    print("  Concrete measurements on your system:")
    print("  ─────────────────────────────────────")

    sizes = [100, 1000, 10000, 100000, 1000000]
    ctx, queue, _ = get_gpu_context()

    KERNEL = """
    __kernel void saxpy(__global const float* X,
                        __global float* Y,
                        const float a, const int N) {
        int i = get_global_id(0);
        if (i < N) Y[i] = a * X[i] + Y[i];
    }
    """
    prg = cl.Program(ctx, KERNEL).build()
    k = cl.Kernel(prg, "saxpy")

    print(f"\n  {'Size':>10s} {'CPU (ms)':>10s} {'GPU (ms)':>10s} {'Winner':>8s}")
    print("  " + "-" * 45)

    for N in sizes:
        X = np.random.randn(N).astype(np.float32)
        Y = np.random.randn(N).astype(np.float32)

        # CPU
        t0 = time.perf_counter()
        for _ in range(10):
            result = 2.0 * X + Y
        cpu_ms = (time.perf_counter() - t0) / 10 * 1000

        # GPU
        mf = cl.mem_flags
        d_X = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        d_Y = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=Y)
        k.set_args(d_X, d_Y, np.float32(2.0), np.int32(N))
        # Warmup
        cl.enqueue_nd_range_kernel(queue, k, (N,), None).wait()
        # Timed (including transfer back)
        t0 = time.perf_counter()
        for _ in range(10):
            cl.enqueue_copy(queue, d_Y, Y)  # re-upload
            cl.enqueue_nd_range_kernel(queue, k, (N,), None).wait()
            cl.enqueue_copy(queue, Y, d_Y).wait()
        gpu_ms = (time.perf_counter() - t0) / 10 * 1000

        winner = "CPU" if cpu_ms < gpu_ms else "GPU"
        print(f"  {N:>10,d} {cpu_ms:>10.3f} {gpu_ms:>10.3f} {winner:>8s}")

    print("""
  KEY INSIGHT:
  ────────────
  • Small data → CPU wins (GPU launch/transfer overhead)
  • Large data → GPU wins (parallel throughput)
  • The crossover point depends on your specific hardware
  • Interview tip: Always profile, never assume!
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 4: HETEROGENEOUS ORCHESTRATION")
    print("  CPU + GPU + NPU Working Together")
    print("=" * 70)

    # Set firmware
    if not os.environ.get("XLNX_VART_FIRMWARE"):
        xclbin = r"C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
        if os.path.exists(xclbin):
            os.environ["XLNX_VART_FIRMWARE"] = xclbin

    ctx, queue, device = get_gpu_context()
    print(f"  GPU: {device.name}")
    print(f"  NPU: {'VitisAIExecutionProvider' in ort.get_available_providers()}")

    demo_pipeline(ctx, queue)
    demo_task_parallel()
    demo_device_selection()

    print("\n" + "=" * 70)
    print("  SESSION 4 COMPLETE — SUMMARY")
    print("=" * 70)
    print("""
  Patterns learned:
  ─────────────────
  10. Pipeline          — CPU→GPU→NPU sequential stages
  11. Task-Parallel     — Independent tasks on different devices
  12. Device Selection  — Decision tree for workload → device mapping

  Architecture principles:
  ────────────────────────
  • CPU: control plane, preprocessing, small/irregular workloads
  • GPU: large data-parallel compute, big-batch ML inference
  • NPU: sustained AI inference, power-efficient, small batch

  Next: Session 5 — Interview questions & coding challenges
""")


if __name__ == "__main__":
    main()
