"""
Session 3: NPU Inference Patterns — ONNX Runtime + VitisAI on XDNA 2
=====================================================================
Learn how to leverage the AMD NPU for efficient AI inference.

Topics:
  1. ONNX model lifecycle: create → optimize → quantize → run
  2. Execution Providers: CPU vs DirectML (GPU) vs VitisAI (NPU)
  3. Pattern: Model partitioning across devices
  4. Pattern: INT8 quantization for NPU
  5. Pattern: Batched inference and throughput optimization
  6. Profiling inference and understanding the compilation cache

Hardware: AMD XDNA 2 NPU (~50 TOPS INT8)
Run: conda run -n ryzen-ai-1.7.1 python tutorials/session3_npu_inference.py
"""

import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper
import os
import tempfile
import time

TEMP_DIR = tempfile.gettempdir()


# ===================================================================
# HELPER: Create ONNX models
# ===================================================================

def create_matmul_relu_model(M, K, N, name="matmul_relu"):
    """Create Y = ReLU(X @ W + B) — a single dense layer."""
    W_data = np.random.randn(K, N).astype(np.float32) * 0.01
    B_data = np.random.randn(N).astype(np.float32) * 0.01

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])
    W = numpy_helper.from_array(W_data, name="W")
    B = numpy_helper.from_array(B_data, name="B")

    matmul = helper.make_node("MatMul", ["X", "W"], ["matmul_out"])
    add = helper.make_node("Add", ["matmul_out", "B"], ["add_out"])
    relu = helper.make_node("Relu", ["add_out"], ["Y"])

    graph = helper.make_graph([matmul, add, relu], name, [X], [Y], [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def create_multi_layer_model(M, hidden, N, num_layers=4, name="mlp"):
    """Create a multi-layer perceptron: X → Dense → ReLU → ... → Y."""
    nodes = []
    initializers = []
    dims = [M] + [hidden] * (num_layers - 1) + [N]

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, dims[0]])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, dims[-1]])

    prev_name = "input"
    for i in range(num_layers):
        w_name = f"W{i}"
        b_name = f"B{i}"
        mm_out = f"mm{i}"
        add_out = f"add{i}"
        relu_out = f"relu{i}" if i < num_layers - 1 else "output"

        W_data = np.random.randn(dims[i], dims[i + 1]).astype(np.float32) * 0.01
        B_data = np.zeros(dims[i + 1], dtype=np.float32)
        initializers.append(numpy_helper.from_array(W_data, name=w_name))
        initializers.append(numpy_helper.from_array(B_data, name=b_name))

        nodes.append(helper.make_node("MatMul", [prev_name, w_name], [mm_out]))
        nodes.append(helper.make_node("Add", [mm_out, b_name], [add_out]))
        if i < num_layers - 1:
            nodes.append(helper.make_node("Relu", [add_out], [relu_out]))
            prev_name = relu_out
        else:
            # Last layer: overwrite add_out to "output"
            nodes[-1] = helper.make_node("Add", [mm_out, b_name], ["output"])

    graph = helper.make_graph(nodes, name, [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def save_model(model, name):
    path = os.path.join(TEMP_DIR, f"{name}.onnx")
    onnx.save(model, path)
    return path


# ===================================================================
# LESSON 1: Execution Provider Comparison
# ===================================================================
# The same ONNX model runs on different hardware via providers.
# Each provider compiles/optimizes the graph for its target.
#
# Interview Q: "How does ONNX Runtime select which ops run where?"
# Answer: Provider priority list. Unimplemented ops fall through
#         to the next provider (usually CPU as fallback).
# ===================================================================

def lesson_ep_comparison():
    """Compare inference speed across CPU, GPU (DirectML), NPU (VitisAI)."""
    print("\n" + "=" * 70)
    print("  LESSON 1: EXECUTION PROVIDER COMPARISON")
    print("  Same model, three targets: CPU / GPU (DirectML) / NPU (VitisAI)")
    print("=" * 70)

    M, K, N = 128, 256, 128
    model = create_matmul_relu_model(M, K, N, "ep_compare")
    path = save_model(model, "ep_compare")

    X = np.random.randn(M, K).astype(np.float32)
    iterations = 50

    providers_to_test = [
        ("CPUExecutionProvider", "CPU (Zen 5)"),
        ("DmlExecutionProvider", "GPU (DirectML)"),
        ("VitisAIExecutionProvider", "NPU (VitisAI)"),
    ]

    available = ort.get_available_providers()
    results = []

    for ep, label in providers_to_test:
        if ep not in available:
            print(f"\n  [{label}] — NOT AVAILABLE")
            continue

        print(f"\n  [{label}]")
        sess = ort.InferenceSession(path, providers=[ep])
        active = sess.get_providers()
        print(f"    Active providers: {active}")

        # Warmup
        for _ in range(10):
            sess.run(None, {"X": X})

        # Timed
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            sess.run(None, {"X": X})
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        print(f"    Average: {avg_ms:.2f} ms (+/- {std_ms:.2f})")
        results.append((label, avg_ms))

    if results:
        baseline = results[0][1]
        print(f"\n  Comparison (vs {results[0][0]}):")
        for label, avg in results:
            print(f"    {label:25s}  {avg:8.2f} ms  ({baseline / avg:.2f}x)")

    os.remove(path)

    print("""
  KEY CONCEPTS:
  ─────────────
  • Provider priority: first provider gets first chance at each op
  • VitisAI compiles the graph to NPU instructions (first run is slow)
  • Subsequent runs use cached compilation (fast)
  • Not all ops are supported by every EP → ops "fall through" to CPU
  • Interview tip: Understand the provider fallback mechanism
""")


# ===================================================================
# LESSON 2: Model Depth and NPU Compilation
# ===================================================================
# Deeper models take longer to compile but benefit more from NPU.
#
# Interview Q: "What's the tradeoff for NPU vs GPU inference?"
# Answer: NPU has compilation overhead but better power efficiency
#         and consistent latency for sustained workloads.
# ===================================================================

def lesson_model_depth():
    """Show how model depth affects NPU compilation and inference."""
    print("\n" + "=" * 70)
    print("  LESSON 2: MODEL DEPTH vs NPU PERFORMANCE")
    print("  Deeper models: more compilation time, more NPU benefit")
    print("=" * 70)

    configs = [
        (1, "1-layer (shallow)"),
        (4, "4-layer (medium)"),
        (8, "8-layer (deep)"),
    ]

    for num_layers, desc in configs:
        print(f"\n  [{desc}]")
        model = create_multi_layer_model(256, 256, 256, num_layers=num_layers)
        path = save_model(model, f"mlp_{num_layers}L")

        X = np.random.randn(1, 256).astype(np.float32)

        for ep, label in [("CPUExecutionProvider", "CPU"),
                          ("VitisAIExecutionProvider", "NPU")]:
            if ep not in ort.get_available_providers():
                continue

            # First run (includes compilation for NPU)
            t0 = time.perf_counter()
            sess = ort.InferenceSession(path, providers=[ep])
            sess.run(None, {"input": X})
            first_run_ms = (time.perf_counter() - t0) * 1000

            # Warm runs
            times = []
            for _ in range(50):
                t0 = time.perf_counter()
                sess.run(None, {"input": X})
                times.append(time.perf_counter() - t0)

            avg_ms = np.mean(times) * 1000
            print(f"    {label:5s}: first={first_run_ms:8.1f} ms  avg={avg_ms:.3f} ms")

        os.remove(path)

    print("""
  KEY CONCEPTS:
  ─────────────
  • NPU first-run includes AIE compiler → can be seconds
  • Compilation results are cached (vaip cache directory)
  • Warm inference on NPU is very fast and power-efficient
  • For serving: pre-warm models during startup
  • Interview tip: NPU shines for sustained, repeated inference
""")


# ===================================================================
# LESSON 3: Batch Size Effects
# ===================================================================
# Interview Q: "How does batch size affect GPU vs NPU performance?"
# Answer: GPU loves large batches (parallel throughput).
#         NPU is optimized for small batch / single inference.
# ===================================================================

def lesson_batch_size():
    """Show how batch size affects performance across providers."""
    print("\n" + "=" * 70)
    print("  LESSON 3: BATCH SIZE EFFECTS")
    print("  GPU: better with large batches. NPU: efficient at batch=1")
    print("=" * 70)

    K, N = 256, 256
    batch_sizes = [1, 4, 16, 64]

    for ep, label in [("CPUExecutionProvider", "CPU"),
                      ("DmlExecutionProvider", "GPU"),
                      ("VitisAIExecutionProvider", "NPU")]:
        if ep not in ort.get_available_providers():
            continue
        print(f"\n  [{label}]")
        for bs in batch_sizes:
            model = create_matmul_relu_model(bs, K, N, f"batch_{bs}")
            path = save_model(model, f"batch_{bs}")

            sess = ort.InferenceSession(path, providers=[ep])
            X = np.random.randn(bs, K).astype(np.float32)

            # Warmup
            for _ in range(10):
                sess.run(None, {"X": X})

            times = []
            for _ in range(30):
                t0 = time.perf_counter()
                sess.run(None, {"X": X})
                times.append(time.perf_counter() - t0)

            avg_ms = np.mean(times) * 1000
            throughput = bs / (np.mean(times))
            print(f"    batch={bs:3d}: {avg_ms:7.3f} ms  ({throughput:.0f} samples/sec)")
            os.remove(path)

    print("""
  KEY CONCEPTS:
  ─────────────
  • CPU: scales linearly with batch size
  • GPU (DirectML): fixed overhead → better efficiency at larger batches
  • NPU: optimized for batch=1 real-time inference
  • Throughput = batch_size / latency — the real metric for serving
  • Interview tip: Match batch size to deployment scenario
    - Edge/real-time → batch=1, NPU preferred
    - Cloud/batch → large batch, GPU preferred
""")


# ===================================================================
# LESSON 4: Multi-Provider Session (Partitioned Inference)
# ===================================================================
# ONNX Runtime can use MULTIPLE providers in one session.
# Ops get assigned to the highest-priority provider that supports them.
#
# Interview Q: "Can you run part of a model on NPU and part on CPU?"
# Answer: Yes — provider priority list. Unsupported ops fall to CPU.
# ===================================================================

def lesson_multi_provider():
    """Demonstrate multi-provider session with fallback."""
    print("\n" + "=" * 70)
    print("  LESSON 4: MULTI-PROVIDER SESSION")
    print("  VitisAI handles supported ops, CPU handles the rest")
    print("=" * 70)

    model = create_multi_layer_model(256, 256, 256, num_layers=4)
    path = save_model(model, "multi_provider")
    X = np.random.randn(1, 256).astype(np.float32)

    configs = [
        (["CPUExecutionProvider"], "CPU only"),
        (["VitisAIExecutionProvider", "CPUExecutionProvider"], "NPU + CPU fallback"),
        (["DmlExecutionProvider", "CPUExecutionProvider"], "GPU + CPU fallback"),
    ]

    for providers, label in configs:
        available = ort.get_available_providers()
        active_providers = [p for p in providers if p in available]
        if not active_providers:
            continue

        print(f"\n  [{label}]")
        sess = ort.InferenceSession(path, providers=active_providers)
        print(f"    Requested: {active_providers}")
        print(f"    Active:    {sess.get_providers()}")

        # Warmup
        for _ in range(10):
            sess.run(None, {"input": X})

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            sess.run(None, {"input": X})
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        result = sess.run(None, {"input": X})[0]
        print(f"    Avg latency: {avg_ms:.3f} ms")
        print(f"    Output shape: {result.shape}")

    os.remove(path)

    print("""
  KEY CONCEPTS:
  ─────────────
  • Provider list = priority order. First match wins per op.
  • VitisAI EP handles MatMul, Conv, etc. → mapped to AIE tiles
  • Unsupported ops (custom, exotic) → fall through to CPU EP
  • This is automatic — no manual partitioning needed
  • Interview tip: Explain the graph partitioning algorithm
    1. Walk graph nodes in topological order
    2. Assign each node to highest-priority EP that claims it
    3. Insert data transfers at EP boundaries
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 3: NPU INFERENCE PATTERNS")
    print("  ONNX Runtime on AMD XDNA 2 — VitisAI Execution Provider")
    print("=" * 70)

    print(f"\n  ONNX Runtime: {ort.__version__}")
    print(f"  Providers:    {ort.get_available_providers()}")

    # Set firmware if not set
    if not os.environ.get("XLNX_VART_FIRMWARE"):
        xclbin = r"C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
        if os.path.exists(xclbin):
            os.environ["XLNX_VART_FIRMWARE"] = xclbin

    lesson_ep_comparison()
    lesson_model_depth()
    lesson_batch_size()
    lesson_multi_provider()

    print("\n" + "=" * 70)
    print("  SESSION 3 COMPLETE — SUMMARY")
    print("=" * 70)
    print("""
  Lessons learned:
  ────────────────
  1. EP Comparison  — Same model, 3 targets (CPU/GPU/NPU)
  2. Model Depth    — NPU compilation cache, first-run vs warm latency
  3. Batch Size     — GPU scales with batch, NPU optimal at batch=1
  4. Multi-Provider — Automatic graph partitioning across providers

  NPU (XDNA 2) sweet spot:
  ─────────────────────────
  • Quantized models (INT8/INT4)
  • Small batch / real-time inference
  • Sustained, power-efficient AI workloads
  • Transformer attention, CNN inference

  Next: Session 4 — Heterogeneous orchestration (CPU + GPU + NPU pipeline)
""")


if __name__ == "__main__":
    main()
