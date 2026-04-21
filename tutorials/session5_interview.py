"""
Session 5: Interview Questions & Coding Challenges
===================================================
Practice the most common parallel computing interview questions.
Each problem includes: question, hints, GPU solution, and analysis.

Topics:
  Q1. Parallel Array Sum (Reduce)
  Q2. Find Maximum Element (Reduce variant)
  Q3. Move Zeros to End (Stream Compaction / Scan)
  Q4. Matrix Multiply Complexity Analysis
  Q5. Producer-Consumer with Bounded Buffer
  Q6. Parallel Merge Sort (Bitonic Sort on GPU)
  Q7. Softmax on GPU (numerically stable)
  Q8. Convolution 2D (tiled with halo)
  Q9. Race Condition Detection
  Q10. System Design: Real-time AI Edge Pipeline

Hardware: AMD Radeon 860M + XDNA 2 NPU
Run: conda run -n ryzen-ai-1.7.1 python tutorials/session5_interview.py
"""

import numpy as np
import pyopencl as cl
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto, numpy_helper
import os
import tempfile
import time

TEMP_DIR = tempfile.gettempdir()


def get_gpu_context():
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                return ctx, queue, device
    raise RuntimeError("No GPU found")


# ===================================================================
# Q1: PARALLEL ARRAY SUM
# ===================================================================
# "Sum an array of N floats on a GPU. What's the time complexity?"
#
# Answer: O(N/P + log P) where P = number of processors
# Sequential: O(N). Parallel: O(N/P) for loading + O(log P) for tree reduce.
# ===================================================================

Q1_KERNEL = """
__kernel void array_sum(__global const float* data,
                        __global float* partials,
                        __local float* scratch,
                        const int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gs = get_local_size(0);

    // Grid-stride loop: handle more elements than work-items
    float acc = 0.0f;
    for (int i = gid; i < N; i += get_global_size(0)) {
        acc += data[i];
    }
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gs / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partials[get_group_id(0)] = scratch[0];
}
"""

def q1_parallel_sum(ctx, queue):
    print("\n" + "=" * 70)
    print("  Q1: PARALLEL ARRAY SUM")
    print("  'Sum N numbers on a GPU. Analyze the complexity.'")
    print("=" * 70)

    N = 1 << 22  # 4M elements
    LS = 256
    num_groups = min(256, (N + LS - 1) // LS)
    data = np.random.randn(N).astype(np.float32)

    mf = cl.mem_flags
    d_data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_partial = cl.Buffer(ctx, mf.WRITE_ONLY, num_groups * 4)

    prg = cl.Program(ctx, Q1_KERNEL).build()
    kernel = cl.Kernel(prg, "array_sum")
    kernel.set_args(d_data, d_partial, cl.LocalMemory(LS * 4), np.int32(N))

    evt = cl.enqueue_nd_range_kernel(queue, kernel, (num_groups * LS,), (LS,))
    evt.wait()
    gpu_ns = evt.profile.end - evt.profile.start

    partial = np.empty(num_groups, dtype=np.float32)
    cl.enqueue_copy(queue, partial, d_partial).wait()
    gpu_sum = np.sum(partial)
    cpu_sum = np.sum(data)

    rel_err = abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-10)
    print(f"\n  N = {N:,}, Work-groups = {num_groups}")
    print(f"  GPU sum = {gpu_sum:.4f}, CPU sum = {cpu_sum:.4f}")
    print(f"  GPU kernel: {gpu_ns / 1e6:.2f} ms, Error: {rel_err:.2e}")
    print(f"  Result: {'PASS' if rel_err < 1e-3 else 'FAIL'}")

    print("""
  SOLUTION ANALYSIS:
  ──────────────────
  Key optimization: Grid-stride loop
    • Each work-item sums MULTIPLE elements (N / global_size)
    • Reduces launch overhead, better for large N
    • Then tree-reduce within each work-group

  Complexity: O(N/P + log(WG_SIZE))
    • N/P: each work-item processes N/global_size elements
    • log(WG_SIZE): tree reduction within work-group
    • P = num_groups × WG_SIZE = total work-items

  Common mistakes:
    • Forgetting barrier() → race condition
    • Not handling N not divisible by work-group size
    • FP32 accumulation order differs → slightly different results
""")


# ===================================================================
# Q2: FIND MAXIMUM (Reduce Variant)
# ===================================================================
# "Find the maximum value AND its index in a parallel array."
# ===================================================================

Q2_KERNEL = """
__kernel void find_max(__global const float* data,
                       __global float* max_vals,
                       __global int* max_idxs,
                       __local float* s_vals,
                       __local int* s_idxs,
                       const int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gs = get_local_size(0);

    float best_val = -INFINITY;
    int best_idx = -1;

    for (int i = gid; i < N; i += get_global_size(0)) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }

    s_vals[lid] = best_val;
    s_idxs[lid] = best_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gs / 2; s > 0; s >>= 1) {
        if (lid < s) {
            if (s_vals[lid + s] > s_vals[lid]) {
                s_vals[lid] = s_vals[lid + s];
                s_idxs[lid] = s_idxs[lid + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        max_vals[get_group_id(0)] = s_vals[0];
        max_idxs[get_group_id(0)] = s_idxs[0];
    }
}
"""

def q2_find_max(ctx, queue):
    print("\n" + "=" * 70)
    print("  Q2: FIND MAXIMUM WITH INDEX")
    print("  'Find max value AND its position in parallel.'")
    print("=" * 70)

    N = 1 << 22
    LS = 256
    num_groups = min(256, (N + LS - 1) // LS)
    data = np.random.randn(N).astype(np.float32)

    mf = cl.mem_flags
    d_data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_vals = cl.Buffer(ctx, mf.WRITE_ONLY, num_groups * 4)
    d_idxs = cl.Buffer(ctx, mf.WRITE_ONLY, num_groups * 4)

    prg = cl.Program(ctx, Q2_KERNEL).build()
    kernel = cl.Kernel(prg, "find_max")
    kernel.set_args(d_data, d_vals, d_idxs,
                    cl.LocalMemory(LS * 4), cl.LocalMemory(LS * 4), np.int32(N))

    evt = cl.enqueue_nd_range_kernel(queue, kernel, (num_groups * LS,), (LS,))
    evt.wait()

    vals = np.empty(num_groups, dtype=np.float32)
    idxs = np.empty(num_groups, dtype=np.int32)
    cl.enqueue_copy(queue, vals, d_vals).wait()
    cl.enqueue_copy(queue, idxs, d_idxs).wait()

    best = np.argmax(vals)
    gpu_max_val = vals[best]
    gpu_max_idx = idxs[best]

    cpu_max_idx = np.argmax(data)
    cpu_max_val = data[cpu_max_idx]

    print(f"\n  N = {N:,}")
    print(f"  GPU: max = {gpu_max_val:.6f} at index {gpu_max_idx}")
    print(f"  CPU: max = {cpu_max_val:.6f} at index {cpu_max_idx}")
    print(f"  Result: {'PASS' if gpu_max_val == cpu_max_val else 'FAIL'}")

    print("""
  SOLUTION ANALYSIS:
  ──────────────────
  • Carry BOTH value and index through the reduction
  • Compare-and-swap: if neighbor is larger, take its value AND index
  • Same tree pattern as sum, but with custom operator
  • Generalizes to: argmin, top-K, k-th largest element
""")


# ===================================================================
# Q3: MOVE ZEROS TO END (Stream Compaction)
# ===================================================================
# "Given array with zeros, move all non-zeros to front. Preserve order."
# This is STREAM COMPACTION — fundamental GPU algorithm.
# ===================================================================

Q3_SCAN_KERNEL = """
// Step 1: Mark non-zero elements (predicate)
__kernel void mark_nonzero(__global const float* data,
                           __global int* flags,
                           const int N) {
    int i = get_global_id(0);
    if (i < N) flags[i] = (data[i] != 0.0f) ? 1 : 0;
}

// Step 2: Prefix sum on flags (gives destination indices)
// Using Hillis-Steele for simplicity (work-group level)
__kernel void scan(__global int* data,
                   __local int* temp,
                   const int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int n = get_local_size(0);

    temp[lid] = (gid < N) ? data[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 1; stride < n; stride <<= 1) {
        int val = (lid >= stride) ? temp[lid - stride] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[lid] += val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < N) data[gid] = temp[lid];
}

// Step 3: Scatter non-zero elements to compacted positions
__kernel void compact(__global const float* input,
                      __global const int* scan_result,
                      __global const int* flags,
                      __global float* output,
                      const int N) {
    int i = get_global_id(0);
    if (i < N && flags[i] == 1) {
        int dest = scan_result[i] - 1;  // Inclusive scan, so subtract 1
        output[dest] = input[i];
    }
}
"""

def q3_move_zeros(ctx, queue):
    print("\n" + "=" * 70)
    print("  Q3: MOVE ZEROS TO END (Stream Compaction)")
    print("  'Remove zeros from array, preserve order. Do it on GPU.'")
    print("=" * 70)

    N = 256  # Keep small for verification
    data = np.array([3, 0, 1, 0, 0, 7, 2, 0, 4, 0, 5, 0, 0, 8, 6, 0] * (N // 16),
                    dtype=np.float32)
    N = len(data)

    mf = cl.mem_flags
    d_data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_flags = cl.Buffer(ctx, mf.READ_WRITE, N * 4)
    d_output = cl.Buffer(ctx, mf.WRITE_ONLY, N * 4)

    prg = cl.Program(ctx, Q3_SCAN_KERNEL).build()

    # Step 1: Mark non-zeros
    k_mark = cl.Kernel(prg, "mark_nonzero")
    k_mark.set_args(d_data, d_flags, np.int32(N))
    cl.enqueue_nd_range_kernel(queue, k_mark, (N,), None).wait()

    # Read flags, copy for scan
    flags = np.empty(N, dtype=np.int32)
    cl.enqueue_copy(queue, flags, d_flags).wait()
    scan_data = flags.copy()
    d_scan = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=scan_data)

    # Step 2: Prefix sum
    k_scan = cl.Kernel(prg, "scan")
    k_scan.set_args(d_scan, cl.LocalMemory(N * 4), np.int32(N))
    cl.enqueue_nd_range_kernel(queue, k_scan, (N,), (N,)).wait()

    cl.enqueue_copy(queue, scan_data, d_scan).wait()

    # Step 3: Compact
    k_compact = cl.Kernel(prg, "compact")
    k_compact.set_args(d_data, d_scan, d_flags, d_output, np.int32(N))
    cl.enqueue_nd_range_kernel(queue, k_compact, (N,), None).wait()

    output = np.zeros(N, dtype=np.float32)
    cl.enqueue_copy(queue, output, d_output).wait()

    num_nonzero = int(scan_data[-1])
    compacted = output[:num_nonzero]

    # CPU reference
    cpu_result = data[data != 0]

    match = np.array_equal(compacted, cpu_result)
    print(f"\n  Input[:16]:   {data[:16].astype(int)}")
    print(f"  Flags[:16]:   {flags[:16]}")
    print(f"  Scan[:16]:    {scan_data[:16]}")
    print(f"  Output[:16]:  {compacted[:16].astype(int)}")
    print(f"  Non-zeros:    {num_nonzero} / {N}")
    print(f"  Result:       {'PASS' if match else 'FAIL'}")

    print("""
  SOLUTION ANALYSIS:
  ──────────────────
  Three-step stream compaction:
    1. Predicate: mark which elements to keep (flag = 0 or 1)
    2. Scan: prefix sum on flags → gives output position for each kept element
    3. Scatter: write kept elements to their computed positions

  This is THE fundamental GPU algorithm pattern. Used in:
    • Radix sort, histogram equalization
    • Particle simulations (remove dead particles)
    • Sparse matrix operations
    • "Filter" operations in databases
  
  Interview tip: Know all three steps and why scan is needed
""")


# ===================================================================
# Q4: NUMERICALLY STABLE SOFTMAX ON GPU
# ===================================================================
# "Implement softmax that won't overflow or underflow."
# softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
# ===================================================================

Q4_KERNEL = """
__kernel void softmax(__global const float* input,
                      __global float* output,
                      __local float* scratch,
                      const int N) {
    int lid = get_local_id(0);
    int gs = get_local_size(0);

    // Step 1: Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = lid; i < N; i += gs) {
        local_max = fmax(local_max, input[i]);
    }
    scratch[lid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gs / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = lid; i < N; i += gs) {
        local_sum += exp(input[i] - max_val);
    }
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = gs / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 3: Normalize
    for (int i = lid; i < N; i += gs) {
        output[i] = exp(input[i] - max_val) / total_sum;
    }
}
"""

def q4_softmax(ctx, queue):
    print("\n" + "=" * 70)
    print("  Q4: NUMERICALLY STABLE SOFTMAX ON GPU")
    print("  'Implement softmax without overflow. Very common interview Q.'")
    print("=" * 70)

    N = 1024
    LS = 256
    data = np.random.randn(N).astype(np.float32) * 10  # Large values to test stability

    mf = cl.mem_flags
    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    prg = cl.Program(ctx, Q4_KERNEL).build()
    kernel = cl.Kernel(prg, "softmax")
    kernel.set_args(d_in, d_out, cl.LocalMemory(LS * 4), np.int32(N))

    evt = cl.enqueue_nd_range_kernel(queue, kernel, (LS,), (LS,))
    evt.wait()

    result = np.empty_like(data)
    cl.enqueue_copy(queue, result, d_out).wait()

    # CPU reference (stable)
    shifted = data - np.max(data)
    exp_vals = np.exp(shifted)
    cpu_result = exp_vals / np.sum(exp_vals)

    max_err = np.max(np.abs(result - cpu_result))
    sum_check = np.sum(result)

    print(f"\n  Vector size:    {N}")
    print(f"  Sum of output:  {sum_check:.6f} (should be 1.0)")
    print(f"  Max error:      {max_err:.2e}")
    print(f"  All positive:   {np.all(result >= 0)}")
    print(f"  Result:         {'PASS' if max_err < 1e-5 and abs(sum_check - 1.0) < 1e-5 else 'FAIL'}")

    print("""
  SOLUTION ANALYSIS:
  ──────────────────
  Three-pass algorithm:
    1. Find max(x) — parallel reduce (max)
    2. Compute sum(exp(x - max)) — fused map + reduce
    3. Normalize: exp(x - max) / sum — parallel map

  Why subtract max?
    • exp(1000) = inf (overflow!)
    • exp(1000 - 1000) = exp(0) = 1 (safe)
    • Mathematically equivalent: softmax(x) = softmax(x - c)

  Interview tip: ALWAYS mention the max-subtraction trick
  Follow-up: "How to do it in one pass?" → Online softmax algorithm
""")


# ===================================================================
# Q5: RACE CONDITION DETECTION
# ===================================================================
# "What's wrong with this code? How to fix it?"
# ===================================================================

def q5_race_condition():
    print("\n" + "=" * 70)
    print("  Q5: RACE CONDITION DETECTION")
    print("  'Find the bug in this parallel code. How to fix it?'")
    print("=" * 70)

    print("""
  BUGGY CODE:
  ═══════════
  __kernel void buggy_sum(__global float* data, __global float* result) {
      int gid = get_global_id(0);
      *result += data[gid];  // BUG: race condition!
  }

  PROBLEM:
  ────────
  Multiple work-items read-modify-write *result simultaneously.
  Read  by thread A: result = 5.0
  Read  by thread B: result = 5.0
  Write by thread A: result = 5.0 + data[A] = 8.0
  Write by thread B: result = 5.0 + data[B] = 7.0  ← OVERWRITES A's update!

  FIXES (3 options):
  ──────────────────
  1. atomic_add(&result, data[gid])  — correct but SLOW (serialized)
  2. Tree reduction in local memory  — FAST (Session 1 Pattern 2)
  3. Per-work-group partial sums     — FASTEST (grid-stride loop)

  INTERVIEW FOLLOW-UPS:
  ─────────────────────
  • "Why not just use atomics?" → Serializes access, O(N) contention
  • "What if the operator isn't associative?" → Can't use tree reduce
  • "How to detect race conditions?" → Thread sanitizers, GPU debuggers
  • "What about GPU memory model?" → Work-items in same WG share LDS,
    barrier() provides acquire/release semantics within a WG.
    Different WGs: NO synchronization (except atomics to global mem).
""")


# ===================================================================
# Q6: SYSTEM DESIGN — REAL-TIME EDGE AI
# ===================================================================
# "Design a real-time video classification system for edge deployment
#  on an AMD Ryzen AI laptop."
# ===================================================================

def q6_system_design():
    print("\n" + "=" * 70)
    print("  Q6: SYSTEM DESIGN — Real-Time Edge AI Pipeline")
    print("  'Design a video classification system on AMD Ryzen AI laptop'")
    print("=" * 70)

    print("""
  REQUIREMENTS:
  • 30 FPS camera input (33ms per frame budget)
  • Object detection + classification
  • < 50ms end-to-end latency
  • Battery-efficient (laptop deployment)

  ARCHITECTURE:
  ═════════════

  Camera (30fps)
    │
    ▼
  ┌─────────────────────────────────────────────┐
  │  CPU Thread: Frame Acquisition              │
  │  • Decode camera frame                      │  ~2ms
  │  • Color space conversion (BGR→RGB)         │
  │  • Ring buffer management                   │
  └────────────────┬────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────┐
  │  GPU (DirectML): Preprocessing              │
  │  • Resize to model input (640×640)          │  ~3ms
  │  • Normalize (mean/std)                     │
  │  • Letterbox padding                        │
  └────────────────┬────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────┐
  │  NPU (VitisAI): Neural Network Inference    │
  │  • YOLOv8-nano INT8 (quantized)             │  ~10ms
  │  • Runs entirely on XDNA 2 AIE tiles        │
  │  • 50 TOPS INT8 throughput                  │
  └────────────────┬────────────────────────────┘
                   │
                   ▼
  ┌─────────────────────────────────────────────┐
  │  CPU Thread: Postprocessing                 │
  │  • NMS (Non-Maximum Suppression)            │  ~1ms
  │  • Tracking (IoU-based)                     │
  │  • Render bounding boxes                    │
  └─────────────────────────────────────────────┘

  TOTAL: ~16ms (< 33ms budget) → 60+ FPS possible!

  KEY DESIGN DECISIONS:
  ─────────────────────
  1. NPU for inference: 10x more power-efficient than GPU
  2. INT8 quantization: full TOPS utilization on XDNA 2
  3. Double-buffering: CPU acquires frame N+1 while NPU runs frame N
  4. GPU for preprocessing: parallel resize/normalize is GPU-friendly
  5. CPU for NMS: irregular, branchy control flow — CPU excels

  POWER BUDGET:
  ─────────────
  • CPU preprocessing: ~2W
  • GPU resize/norm:   ~3W
  • NPU inference:     ~5W  (vs ~15W on GPU!)
  • Total:             ~10W  (vs ~20W all-GPU approach)

  INTERVIEW TIPS:
  ───────────────
  • Start with the latency budget (33ms at 30fps)
  • Justify each device choice
  • Mention double-buffering for overlap
  • Discuss fallback: what if NPU doesn't support an op?
  • Scaling: multiple camera streams → batch on NPU
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 5: INTERVIEW QUESTIONS & CODING CHALLENGES")
    print("  GPU + NPU Parallel Programming")
    print("=" * 70)

    ctx, queue, device = get_gpu_context()
    print(f"  GPU: {device.name}")

    q1_parallel_sum(ctx, queue)
    q2_find_max(ctx, queue)
    q3_move_zeros(ctx, queue)
    q4_softmax(ctx, queue)
    q5_race_condition()
    q6_system_design()

    print("\n" + "=" * 70)
    print("  SESSION 5 COMPLETE — INTERVIEW CHEAT SHEET")
    print("=" * 70)
    print("""
  Pattern Quick Reference:
  ════════════════════════
  Map        → get_global_id, no sync           → activation, normalize
  Reduce     → tree in local mem + barrier       → sum, max, dot product
  Scan       → prefix sum (Hillis-Steele/Blelloch) → sort, compact, histogram
  Stencil    → halo cells + local mem            → convolution, blur, PDE
  Histogram  → local atomics → global merge      → binning, statistics
  Transpose  → tiled through LDS + padding       → memory layout transform
  Scatter    → indexed write (needs atomics)      → avoid if possible
  Gather     → indexed read (safe)                → permutation, lookup
  Compact    → predicate + scan + scatter         → filter, remove elements
  Softmax    → max-reduce + exp-sum-reduce + map  → neural net output layer

  Key Concepts to Mention:
  ════════════════════════
  • Memory coalescing (adjacent threads → adjacent addresses)
  • Bank conflicts in local memory (TILE+1 padding trick)
  • Work-group size selection (multiple of wavefront size: 32 or 64)
  • Grid-stride loops (handle arbitrary N with fixed work-items)
  • Kernel fusion (reduce memory passes)
  • Provider fallback in ONNX Runtime
  • NPU compilation cache and warm-up latency
  • Device selection decision tree (data size, parallelism, power)

  Top Interview Mistakes:
  ═══════════════════════
  ✗ Forgetting barrier() after local memory operations
  ✗ Race condition on global memory (use atomics or reduce)
  ✗ Assuming GPU is always faster (small data → CPU wins)
  ✗ Ignoring data transfer overhead (PCIe / memory copy)
  ✗ Not handling edge cases (N not divisible by work-group size)
""")


if __name__ == "__main__":
    main()
