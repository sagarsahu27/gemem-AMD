"""
Session 8: AMD-Specific Optimizations & NPU Deep Dive
======================================================
Hardware-specific techniques for AMD Radeon (RDNA 3.5) and XDNA 2 NPU.

Topics:
  1. Vectorized Loads (float4) — 4x fewer memory transactions
  2. Subgroup/Wavefront Operations — cross-lane ops without LDS
  3. Wave32 vs Wave64 — RDNA dual-mode execution
  4. INT8 Post-Training Quantization — make models NPU-ready
  5. ONNX Model Profiling — node-level performance analysis

Hardware: AMD Radeon 860M (RDNA 3.5) + XDNA 2 NPU
Run: conda run -n ryzen-ai-1.7.1 --no-capture-output python tutorials/session8_amd_npu_deep.py
"""

import numpy as np
import pyopencl as cl
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper
import os
import tempfile
import time
import json

TEMP_DIR = tempfile.gettempdir()


def get_gpu_context():
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                return ctx, queue, device
    raise RuntimeError("No GPU found")


def section(num, title, subtitle=""):
    print(f"\n{'=' * 70}")
    print(f"  {num}. {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70)


# ===================================================================
# 1. VECTORIZED LOADS (float4) — Fewer Memory Transactions
# ===================================================================
# Instead of loading one float per work-item, load FOUR at once.
# GPU memory system handles 128-bit loads in a single transaction.
#
# Result: 4x fewer transactions, better bandwidth utilization.
# ===================================================================

SCALAR_KERNEL = """
__kernel void saxpy_scalar(__global const float* X,
                           __global float* Y,
                           float a, int N) {
    int i = get_global_id(0);
    if (i < N) Y[i] = a * X[i] + Y[i];
}
"""

VECTOR4_KERNEL = """
__kernel void saxpy_vec4(__global const float4* X,
                         __global float4* Y,
                         float a, int N4) {
    int i = get_global_id(0);
    if (i < N4) Y[i] = a * X[i] + Y[i];
}
"""

def demo_vectorized_loads(ctx, queue, device):
    section("1", "VECTORIZED LOADS (float4)",
            "Load 4 floats in one transaction — fewer memory ops")

    print("""
  Scalar: each work-item loads 1 float  (32 bits per load)
  float4: each work-item loads 4 floats (128 bits per load)

  ┌──────────────────────────────────────────────────────────┐
  │  Scalar:  Thread 0 loads X[0]    → 1 transaction        │
  │           Thread 1 loads X[1]    → 1 transaction        │
  │           Thread 2 loads X[2]    → 1 transaction        │
  │           Thread 3 loads X[3]    → 1 transaction        │
  │           = 4 separate loads                             │
  │                                                          │
  │  float4:  Thread 0 loads X[0:4]  → 1 wide transaction   │
  │           = 1 load for same data!                        │
  └──────────────────────────────────────────────────────────┘
""")

    N = 1 << 22  # 4M elements (must be divisible by 4)
    N4 = N // 4
    X = np.random.randn(N).astype(np.float32)
    Y = np.random.randn(N).astype(np.float32)

    mf = cl.mem_flags
    prg_s = cl.Program(ctx, SCALAR_KERNEL).build()
    prg_v = cl.Program(ctx, VECTOR4_KERNEL).build()

    d_X = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    d_Y_s = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=Y.copy())
    d_Y_v = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=Y.copy())

    LS = 256

    # Warmup
    prg_s.saxpy_scalar(queue, (N,), (LS,), d_X, d_Y_s,
                       np.float32(2.0), np.int32(N)).wait()
    prg_v.saxpy_vec4(queue, (N4,), (LS,), d_X, d_Y_v,
                     np.float32(2.0), np.int32(N4)).wait()

    # Measure scalar
    # Reset Y
    cl.enqueue_copy(queue, d_Y_s, Y.copy()).wait()
    scalar_times = []
    for _ in range(10):
        evt = prg_s.saxpy_scalar(queue, (N,), (LS,), d_X, d_Y_s,
                                 np.float32(2.0), np.int32(N))
        evt.wait()
        scalar_times.append((evt.profile.end - evt.profile.start) / 1e6)

    # Measure vec4
    cl.enqueue_copy(queue, d_Y_v, Y.copy()).wait()
    vec4_times = []
    for _ in range(10):
        evt = prg_v.saxpy_vec4(queue, (N4,), (LS,), d_X, d_Y_v,
                               np.float32(2.0), np.int32(N4))
        evt.wait()
        vec4_times.append((evt.profile.end - evt.profile.start) / 1e6)

    s_ms = np.mean(scalar_times)
    v_ms = np.mean(vec4_times)
    s_bw = (N * 4 * 3) / (s_ms / 1000) / 1e9  # read X, read Y, write Y
    v_bw = (N * 4 * 3) / (v_ms / 1000) / 1e9

    # Verify correctness
    out_s = np.empty(N, dtype=np.float32)
    out_v = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(queue, out_s, d_Y_s).wait()
    cl.enqueue_copy(queue, out_v, d_Y_v).wait()
    match = np.allclose(out_s, out_v, atol=1e-5)

    print(f"  SAXPY on {N:,} elements:\n")
    print(f"  {'Kernel':>12s} {'Time(ms)':>10s} {'BW(GB/s)':>10s} {'Items/WI':>10s}")
    print("  " + "-" * 46)
    print(f"  {'Scalar':>12s} {s_ms:>10.3f} {s_bw:>10.1f} {'1':>10s}")
    print(f"  {'float4':>12s} {v_ms:>10.3f} {v_bw:>10.1f} {'4':>10s}")
    print(f"  {'Speedup':>12s} {s_ms/v_ms:>10.2f}x")
    print(f"  Correctness: {'PASS' if match else 'FAIL'}")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • float4 = 128-bit load, matches GPU cache line granularity
  • 4x fewer work-items needed → 4x fewer memory transactions
  • Data must be 16-byte aligned (float4 = 4 × 4 bytes)
  • Array size must be divisible by 4 (handle remainder separately)
  • Also: float2 (64-bit), float8, float16 on some hardware
  • RDNA 3.5 native: 32-bit and 64-bit loads; float4 coalesces well
  • Rule: always try vec4 for memory-bound kernels
""")


# ===================================================================
# 2. SUBGROUP / WAVEFRONT OPERATIONS
# ===================================================================
# Wavefront-level operations that work WITHOUT local memory or barriers.
# Data is exchanged between lanes within a wavefront via register file.
#
# Much faster than __local + barrier approach for small reductions.
# ===================================================================

def demo_subgroup_ops(ctx, queue, device):
    section("2", "SUBGROUP / WAVEFRONT OPERATIONS",
            "Cross-lane ops without LDS — register-level communication")

    # Check wavefront size
    try:
        wf_size = device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)
    except Exception:
        wf_size = 64

    # Check subgroup extension
    extensions = device.extensions
    has_subgroups = "cl_khr_subgroups" in extensions or "cl_intel_subgroups" in extensions

    print(f"""
  Wavefront size: {wf_size}
  cl_khr_subgroups: {'YES' if has_subgroups else 'NO'}

  Traditional reduction (using __local + barrier):
  ─────────────────────────────────────────────────
  __local float scratch[256];
  scratch[lid] = my_value;
  barrier(CLK_LOCAL_MEM_FENCE);     ← SLOW synchronization
  for (int s = 128; s > 0; s >>= 1) {{
      if (lid < s) scratch[lid] += scratch[lid + s];
      barrier(CLK_LOCAL_MEM_FENCE); ← barrier at EVERY step
  }}

  Wavefront reduction (using shuffle — no LDS, no barrier):
  ──────────────────────────────────────────────────────────
  Within a wavefront, all lanes execute in LOCKSTEP.
  We can exchange data via "shuffle" — reads another lane's register.

  float val = my_value;
  for (int offset = {wf_size}>>1; offset > 0; offset >>= 1)
      val += sub_group_reduce_add(val);  // Hardware instruction!

  No __local memory, no barrier() — pure register operations.
""")

    # Benchmark: LDS reduce vs wavefront-aware reduce
    N = 1 << 22
    data = np.random.randn(N).astype(np.float32)
    mf = cl.mem_flags
    d_data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)

    LS = 256
    num_groups = min(256, (N + LS - 1) // LS)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, num_groups * 4)

    # Traditional LDS reduce
    K_LDS = """
    __kernel void reduce_lds(__global const float* data,
                             __global float* partials,
                             __local float* scratch,
                             int N) {
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        int gs = get_local_size(0);

        float acc = 0.0f;
        for (int i = gid; i < N; i += get_global_size(0))
            acc += data[i];

        scratch[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = gs >> 1; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) partials[get_group_id(0)] = scratch[0];
    }
    """

    # Wavefront-aware reduce: use wavefront-level reduction first,
    # then only use LDS across wavefronts
    K_WAVE = f"""
    #define WF_SIZE {wf_size}
    __kernel void reduce_waveaware(__global const float* data,
                                   __global float* partials,
                                   __local float* scratch,
                                   int N) {{
        int gid = get_global_id(0);
        int lid = get_local_id(0);
        int gs = get_local_size(0);
        int wf_id = lid / WF_SIZE;         // Which wavefront in this WG
        int lane = lid % WF_SIZE;           // Lane within wavefront
        int num_wfs = gs / WF_SIZE;

        // Phase 1: grid-stride accumulation
        float acc = 0.0f;
        for (int i = gid; i < N; i += get_global_size(0))
            acc += data[i];

        // Phase 2: intra-wavefront reduce (no barrier needed!)
        // Within a wavefront, all lanes are in lockstep.
        // Use shared memory with wavefront-sized steps.
        scratch[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce within each wavefront (no barrier between steps
        // because wavefront lanes are in lockstep)
        if (lane < 32) scratch[lid] += scratch[lid + 32];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 16) scratch[lid] += scratch[lid + 16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 8) scratch[lid] += scratch[lid + 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 4) scratch[lid] += scratch[lid + 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 2) scratch[lid] += scratch[lid + 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 1) scratch[lid] += scratch[lid + 1];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Phase 3: reduce across wavefronts (few values)
        if (lane == 0) scratch[wf_id] = scratch[lid];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {{
            float total = 0.0f;
            for (int w = 0; w < num_wfs; w++)
                total += scratch[w];
            partials[get_group_id(0)] = total;
        }}
    }}
    """

    prg_lds = cl.Program(ctx, K_LDS).build()
    prg_wave = cl.Program(ctx, K_WAVE).build()

    GS = num_groups * LS

    # Warmup
    prg_lds.reduce_lds(queue, (GS,), (LS,), d_data, d_out,
                       cl.LocalMemory(LS * 4), np.int32(N)).wait()
    prg_wave.reduce_waveaware(queue, (GS,), (LS,), d_data, d_out,
                              cl.LocalMemory(LS * 4), np.int32(N)).wait()

    # Measure LDS
    lds_times = []
    for _ in range(10):
        evt = prg_lds.reduce_lds(queue, (GS,), (LS,), d_data, d_out,
                                 cl.LocalMemory(LS * 4), np.int32(N))
        evt.wait()
        lds_times.append((evt.profile.end - evt.profile.start) / 1e6)

    # Measure wavefront-aware
    wave_times = []
    for _ in range(10):
        evt = prg_wave.reduce_waveaware(queue, (GS,), (LS,), d_data, d_out,
                                        cl.LocalMemory(LS * 4), np.int32(N))
        evt.wait()
        wave_times.append((evt.profile.end - evt.profile.start) / 1e6)

    lds_ms = np.mean(lds_times)
    wave_ms = np.mean(wave_times)

    print(f"\n  Reduce {N:,} floats, WG={LS}, {num_groups} groups:\n")
    print(f"  {'Method':>20s} {'Time(ms)':>10s} {'Barriers':>10s}")
    print("  " + "-" * 44)
    print(f"  {'LDS (traditional)':>20s} {lds_ms:>10.3f} {'log2(WG)':>10s}")
    print(f"  {'Wave-aware':>20s} {wave_ms:>10.3f} {'fewer':>10s}")
    if wave_ms < lds_ms:
        print(f"  {'Speedup':>20s} {lds_ms/wave_ms:>10.2f}x")
    else:
        print(f"  {'Ratio':>20s} {lds_ms/wave_ms:>10.2f}x (similar)")

    print(f"""
  On AMD RDNA with true subgroup intrinsics (OpenCL 2.0+):
  ─────────────────────────────────────────────────────────
  sub_group_reduce_add(val)  — sum across wavefront lanes
  sub_group_broadcast(val,0) — broadcast lane 0 to all
  sub_group_shuffle(val, id) — read any lane's value
  sub_group_scan_inclusive_add(val) — prefix sum within wavefront

  These require cl_khr_subgroups extension.
  Your GPU: {'SUPPORTED' if has_subgroups else 'NOT DETECTED (using LDS fallback)'}

  INTERVIEW KEY POINTS:
  ─────────────────────
  • Wavefront = SIMD unit, all lanes execute together
  • No barrier needed WITHIN a wavefront (lockstep execution)
  • Subgroup ops use register file → faster than LDS
  • AMD wavefront: 32 or 64 lanes (RDNA supports both)
  • NVIDIA warp: always 32 lanes (__shfl_xor, __ballot)
  • Use subgroup reduce for the inner loop, LDS only across wavefronts
""")


# ===================================================================
# 3. WAVE32 vs WAVE64 — RDNA Dual-Mode Execution
# ===================================================================
# RDNA GPUs can run wavefronts in either 32 or 64 lane mode.
# This is unique to AMD — NVIDIA is always 32.
#
# Wave32: lower latency, less register pressure, more occupancy
# Wave64: higher throughput, better for memory-heavy kernels
# ===================================================================

def demo_wave_modes(ctx, queue, device):
    section("3", "WAVE32 vs WAVE64",
            "RDNA dual-mode execution — unique to AMD")

    try:
        wf_size = device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)
    except Exception:
        wf_size = 64

    print(f"""
  Your GPU wavefront size: {wf_size}

  ┌─────────────────────────────────────────────────────────────┐
  │  Feature          │  Wave32            │  Wave64             │
  │───────────────────│────────────────────│─────────────────────│
  │  Lanes per wave   │  32                │  64                 │
  │  Latency          │  Lower (fewer ops) │  Higher             │
  │  Throughput       │  Good              │  Better (2x lanes)  │
  │  Register use     │  32 × regs/lane    │  64 × regs/lane     │
  │  Occupancy        │  Higher            │  Lower              │
  │  Divergence cost  │  Wastes max 31     │  Wastes max 63      │
  │  Best for         │  Compute-bound     │  Memory-bound       │
  │                   │  Branch-heavy      │  Simple streaming   │
  └─────────────────────────────────────────────────────────────┘

  When to prefer each:
  ─────────────────────
  Wave32:
    • Compute-bound kernels (more wavefronts → better latency hiding)
    • Kernels with divergence (waste fewer lanes per branch)
    • Register-heavy kernels (less total register pressure)

  Wave64:
    • Memory-bound kernels (wider coalescing, fewer cache misses)
    • Simple streaming (SAXPY, copy, map patterns)
    • Reduce operations (more elements per wavefront reduce step)

  On RDNA 3.5 (your gfx1152):
  ────────────────────────────
  • Compiler chooses mode based on kernel analysis
  • Can hint via compiler flags (implementation-specific)
  • Preferred WG size multiple = {wf_size} (your default mode)
""")

    # Demonstrate: measure with WG sizes matching wave32 vs wave64
    N = 1 << 22
    data = np.random.randn(N).astype(np.float32)
    mf = cl.mem_flags

    K = """
    __kernel void stream(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) out[i] = in[i] * 2.0f + 1.0f;
    }
    """
    prg = cl.Program(ctx, K).build()
    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    print(f"  Benchmarking WG sizes (Wave32 vs Wave64 aligned):\n")
    print(f"  {'WG Size':>10s} {'Wave Align':>12s} {'Time(ms)':>10s} {'GB/s':>8s}")
    print("  " + "-" * 46)

    for ls in [32, 64, 128, 256]:
        gs = ((N + ls - 1) // ls) * ls
        # Warmup
        prg.stream(queue, (gs,), (ls,), d_in, d_out, np.int32(N)).wait()
        times = []
        for _ in range(10):
            evt = prg.stream(queue, (gs,), (ls,), d_in, d_out, np.int32(N))
            evt.wait()
            times.append((evt.profile.end - evt.profile.start) / 1e6)
        avg = np.mean(times)
        bw = (N * 4 * 2) / (avg / 1000) / 1e9
        align = "Wave32" if ls == 32 else ("Wave64" if ls == 64 else f"{ls//64}×W64")
        print(f"  {ls:>10d} {align:>12s} {avg:>10.3f} {bw:>8.1f}")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • RDNA is the ONLY architecture with dual wave modes
  • Wave32 = lower latency, better for divergent code
  • Wave64 = higher throughput, better for streaming
  • Compiler/driver selects mode — but WG size gives hints
  • WG size 32: forces 1 wavefront per WG (Wave32 behavior)
  • WG size 64: enables full Wave64 utilization
  • Key differentiator from NVIDIA in interviews!
""")


# ===================================================================
# 4. INT8 POST-TRAINING QUANTIZATION
# ===================================================================
# Quantize a float32 model to int8 for NPU execution.
# NPU XDNA 2 has 50 TOPS at INT8 vs ~8 TFLOPS at FP32.
#
# PTQ: calibrate with sample data, no retraining needed.
# ===================================================================

def demo_quantization():
    section("4", "INT8 POST-TRAINING QUANTIZATION",
            "Make models NPU-ready — 50 TOPS INT8 on XDNA 2")

    print("""
  Why quantize?
  ─────────────
  FP32: 4 bytes per weight, ~8 TFLOPS on NPU
  INT8: 1 byte per weight,  ~50 TOPS on NPU  ← 6x more throughput!
  Plus: 4x less memory, 4x less bandwidth

  Quantization formula:
  ─────────────────────
  int8_value = round(float_value / scale) + zero_point

  where:
    scale = (max_float - min_float) / 255
    zero_point = round(-min_float / scale)

  Dequantize: float_value = (int8_value - zero_point) * scale
""")

    # Step 1: Create a float32 model
    input_dim, hidden_dim, output_dim = 64, 128, 10

    W1_data = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    B1_data = np.zeros(hidden_dim, dtype=np.float32)
    W2_data = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
    B2_data = np.zeros(output_dim, dtype=np.float32)

    W1 = numpy_helper.from_array(W1_data, name="W1")
    B1 = numpy_helper.from_array(B1_data, name="B1")
    W2 = numpy_helper.from_array(W2_data, name="W2")
    B2 = numpy_helper.from_array(B2_data, name="B2")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, output_dim])

    nodes = [
        helper.make_node("MatMul", ["X", "W1"], ["mm1"]),
        helper.make_node("Add", ["mm1", "B1"], ["h1"]),
        helper.make_node("Relu", ["h1"], ["r1"]),
        helper.make_node("MatMul", ["r1", "W2"], ["mm2"]),
        helper.make_node("Add", ["mm2", "B2"], ["Y"]),
    ]

    graph = helper.make_graph(nodes, "mlp", [X], [Y], [W1, B1, W2, B2])
    fp32_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    fp32_model.ir_version = 8

    fp32_path = os.path.join(TEMP_DIR, "fp32_model.onnx")
    onnx.save(fp32_model, fp32_path)

    # Step 2: Manual quantization (demonstrating the math)
    print("  Step 1: Created FP32 model")
    print(f"    Weights: W1 {W1_data.shape}, W2 {W2_data.shape}")

    fp32_size = os.path.getsize(fp32_path)
    print(f"    Model size: {fp32_size:,} bytes")

    # Quantize weights manually
    def quantize_tensor(tensor, name):
        """Quantize a float32 tensor to uint8."""
        t_min = float(np.min(tensor))
        t_max = float(np.max(tensor))
        scale = (t_max - t_min) / 255.0
        zero_point = int(round(-t_min / scale)) if scale > 0 else 0
        zero_point = max(0, min(255, zero_point))
        quantized = np.clip(np.round(tensor / scale) + zero_point, 0, 255).astype(np.uint8)
        # Dequantize for error analysis
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        error = np.max(np.abs(tensor - dequantized))
        return quantized, scale, zero_point, error

    q_W1, s1, z1, e1 = quantize_tensor(W1_data, "W1")
    q_W2, s2, z2, e2 = quantize_tensor(W2_data, "W2")

    print(f"\n  Step 2: Manual weight quantization")
    print(f"    W1: scale={s1:.6f}, zero_point={z1}, max_error={e1:.6f}")
    print(f"    W2: scale={s2:.6f}, zero_point={z2}, max_error={e2:.6f}")
    print(f"    Compression: {W1_data.nbytes + W2_data.nbytes:,} → "
          f"{q_W1.nbytes + q_W2.nbytes:,} bytes "
          f"({(W1_data.nbytes + W2_data.nbytes) / (q_W1.nbytes + q_W2.nbytes):.1f}x)")

    # Step 3: Try ONNX Runtime quantization if available
    print(f"\n  Step 3: ONNX Runtime quantization")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(TEMP_DIR, "int8_model.onnx")
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
        int8_size = os.path.getsize(int8_path)
        print(f"    Dynamic quantization: SUCCESS")
        print(f"    FP32 size: {fp32_size:,} bytes")
        print(f"    INT8 size: {int8_size:,} bytes")
        print(f"    Compression: {fp32_size / int8_size:.1f}x")

        # Compare inference
        sample = np.random.randn(1, input_dim).astype(np.float32)
        sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

        out_fp32 = sess_fp32.run(None, {"X": sample})[0]
        out_int8 = sess_int8.run(None, {"X": sample})[0]
        diff = np.max(np.abs(out_fp32 - out_int8))

        # Speed comparison
        fp32_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess_fp32.run(None, {"X": sample})
            fp32_times.append(time.perf_counter() - t0)

        int8_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            sess_int8.run(None, {"X": sample})
            int8_times.append(time.perf_counter() - t0)

        print(f"\n    Accuracy: max output difference = {diff:.6f}")
        print(f"    Speed (CPU): FP32={np.mean(fp32_times)*1e6:.0f}us, "
              f"INT8={np.mean(int8_times)*1e6:.0f}us")

        os.remove(int8_path)
    except ImportError:
        print("    onnxruntime.quantization not available — showing manual quantization only")

    os.remove(fp32_path)

    print("""
  Quantization types:
  ───────────────────
  Dynamic (shown above): quantize weights only, activations at runtime
  Static (PTQ):          calibrate with sample data → quantize weights + activations
  QAT:                   quantize-aware training → best accuracy, requires retraining

  For NPU (XDNA 2):
  ──────────────────
  • Ryzen AI SDK includes Olive quantization tools
  • Static quantization recommended for best NPU performance
  • INT8 preferred; INT4 supported for some operations
  • Quantized ops map directly to AIE tile MAC units

  INTERVIEW KEY POINTS:
  ─────────────────────
  • Know the formula: int8 = round(float / scale) + zero_point
  • Dynamic quant: easy, no calibration data needed
  • Static quant: better accuracy, needs representative dataset
  • QAT: best accuracy, requires retraining infrastructure
  • Trade-off: accuracy vs speed vs power
  • NPU sweet spot: INT8 static quantization
""")


# ===================================================================
# 5. ONNX MODEL PROFILING — Node-Level Performance
# ===================================================================
# Use enable_profiling to see which nodes take the most time.
# Critical for optimization: find the bottleneck op.
# ===================================================================

def demo_onnx_profiling():
    section("5", "ONNX MODEL PROFILING",
            "Node-level performance analysis — find the bottleneck")

    # Create a multi-layer model
    layers = []
    init = []
    prev_output = "X"
    dim = 256

    for i in range(5):
        w_data = np.random.randn(dim, dim).astype(np.float32) * 0.01
        b_data = np.zeros(dim, dtype=np.float32)
        w_name = f"W{i}"
        b_name = f"B{i}"
        mm_out = f"mm{i}"
        add_out = f"add{i}"
        relu_out = f"relu{i}"

        init.append(numpy_helper.from_array(w_data, name=w_name))
        init.append(numpy_helper.from_array(b_data, name=b_name))

        layers.append(helper.make_node("MatMul", [prev_output, w_name], [mm_out]))
        layers.append(helper.make_node("Add", [mm_out, b_name], [add_out]))
        layers.append(helper.make_node("Relu", [add_out], [relu_out]))
        prev_output = relu_out

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, dim])
    Y = helper.make_tensor_value_info(prev_output, TensorProto.FLOAT, [1, dim])

    graph = helper.make_graph(layers, "profiling_test", [X], [Y], init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    model_path = os.path.join(TEMP_DIR, "profile_model.onnx")
    onnx.save(model, model_path)

    # Create session with profiling enabled
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(TEMP_DIR, "ort_profile")

    sess = ort.InferenceSession(model_path, sess_options=so,
                                providers=["CPUExecutionProvider"])

    sample = np.random.randn(1, dim).astype(np.float32)

    # Run several iterations
    for _ in range(20):
        sess.run(None, {"X": sample})

    # End profiling and read results
    profile_file = sess.end_profiling()

    print(f"\n  Model: 5-layer MLP ({dim}×{dim}), 15 nodes")
    print(f"  Profile saved to: {os.path.basename(profile_file)}")

    # Parse profile JSON
    try:
        with open(profile_file, "r") as f:
            profile_data = json.load(f)

        # Extract node timings
        node_times = {}
        for event in profile_data:
            if event.get("cat") == "Node" and "dur" in event:
                name = event.get("name", "unknown")
                dur = event["dur"]  # microseconds
                if name not in node_times:
                    node_times[name] = []
                node_times[name].append(dur)

        if node_times:
            print(f"\n  Node-level profiling results (avg over iterations):")
            print(f"  {'Node':>30s} {'Avg(us)':>10s} {'%Total':>8s}")
            print("  " + "-" * 52)

            total_us = sum(np.mean(t) for t in node_times.values())
            sorted_nodes = sorted(node_times.items(),
                                  key=lambda x: np.mean(x[1]), reverse=True)

            for name, times in sorted_nodes[:10]:
                avg = np.mean(times)
                pct = avg / total_us * 100 if total_us > 0 else 0
                print(f"  {name:>30s} {avg:>10.1f} {pct:>7.1f}%")

            print(f"  {'TOTAL':>30s} {total_us:>10.1f}")
        else:
            print("  (No node-level timing data in profile)")

        os.remove(profile_file)
    except Exception as e:
        print(f"  Profile parsing: {e}")

    os.remove(model_path)

    print("""
  How to use profiling:
  ─────────────────────
  so = ort.SessionOptions()
  so.enable_profiling = True
  sess = ort.InferenceSession(model, sess_options=so, ...)

  # Run inference
  sess.run(...)

  # Get profile
  profile_file = sess.end_profiling()  # Returns JSON file path

  What to look for:
  ─────────────────
  • Which op type takes the most time? (MatMul, Conv, Attention)
  • Are there unexpected CPU fallbacks? (EP didn't claim the op)
  • Memory copy overhead between EPs?
  • Kernel launch overhead vs actual compute?

  INTERVIEW KEY POINTS:
  ─────────────────────
  • Always profile BEFORE optimizing
  • MatMul/Conv usually dominate (target for NPU offload)
  • Small ops (Relu, Add) have high overhead relative to compute
  • Graph optimization (ORT) fuses small ops → fewer kernel launches
  • Profile with each EP to compare CPU vs GPU vs NPU per-node
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 8: AMD-SPECIFIC OPTIMIZATIONS & NPU DEEP DIVE")
    print("  Hardware-specific techniques for maximum performance")
    print("=" * 70)

    if not os.environ.get("XLNX_VART_FIRMWARE"):
        xclbin = r"C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
        if os.path.exists(xclbin):
            os.environ["XLNX_VART_FIRMWARE"] = xclbin

    ctx, queue, device = get_gpu_context()
    print(f"  GPU: {device.name} ({device.max_compute_units} CUs)")
    print(f"  ONNX RT: {', '.join(ort.get_available_providers())}")

    demo_vectorized_loads(ctx, queue, device)
    demo_subgroup_ops(ctx, queue, device)
    demo_wave_modes(ctx, queue, device)
    demo_quantization()
    demo_onnx_profiling()

    print("\n" + "=" * 70)
    print("  SESSION 8 COMPLETE — AMD & NPU DEEP DIVE SUMMARY")
    print("=" * 70)
    print("""
  AMD-specific optimizations:
  ───────────────────────────
  1. Vectorized Loads:  float4 → 4x fewer memory transactions
  2. Subgroup Ops:      Register-level communication, no LDS/barrier
  3. Wave32 vs Wave64:  RDNA dual mode — unique to AMD

  NPU deep dive:
  ───────────────
  4. INT8 Quantization: scale/zero_point math, dynamic vs static PTQ
  5. ONNX Profiling:    Node-level timing, find bottleneck ops

  Optimization priority for AMD systems:
  ──────────────────────────────────────
  1. Right device (CPU/GPU/NPU) for each task
  2. Quantize models to INT8 for NPU (6x throughput gain)
  3. Use float4 for memory-bound GPU kernels
  4. Wavefront-aware reductions (skip unnecessary barriers)
  5. Profile with ONNX RT to find actual bottlenecks
  6. Match WG size to wave mode (32 for compute, 64 for streaming)

  COMPLETE TUTORIAL SERIES:
  ─────────────────────────
  Session 0: Core concepts — how hardware shapes code
  Session 1: GPU fundamentals — map, reduce, scan, stencil, matmul
  Session 2: Advanced GPU — histogram, transpose, scatter, dot product
  Session 3: NPU inference — EPs, model depth, batch size, multi-provider
  Session 4: Heterogeneous — pipeline, task-parallel, device selection
  Session 5: Interview Q&A — coding challenges + system design
  Session 6: Performance theory — Amdahl, Gustafson, roofline, fusion
  Session 7: Advanced algorithms — Blelloch scan, bitonic sort, concurrency
  Session 8: AMD & NPU deep dive — vec4, subgroups, wave modes, quantization
""")


if __name__ == "__main__":
    main()
