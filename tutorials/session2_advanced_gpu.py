"""
Session 2: Advanced GPU Patterns — Histogram, Transpose, Atomics
================================================================
Build on Session 1 with more complex parallel patterns that appear
in interviews and real-world GPU programming.

Topics:
  1. Pattern: Histogram (atomic operations)
  2. Pattern: Matrix Transpose (coalesced memory access)
  3. Pattern: Scatter / Gather (indirect addressing)
  4. Pattern: Producer-Consumer (pipeline with events)
  5. Memory coalescing and bank conflicts explained

Hardware: AMD Radeon 860M (RDNA 3.5, 4 CUs, OpenCL 2.0)
Run: conda run -n ryzen-ai-1.7.1 python tutorials/session2_advanced_gpu.py
"""

import numpy as np
import pyopencl as cl
import time


def get_gpu_context():
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                print(f"  GPU: {device.name} ({device.max_compute_units} CUs)")
                return ctx, queue, device
    raise RuntimeError("No GPU found")


# ===================================================================
# PATTERN 6: HISTOGRAM (Atomics)
# ===================================================================
# Count occurrences of each value. Requires atomic operations because
# multiple work-items may increment the same bin simultaneously.
#
# Interview Q: "Build a histogram of 10M values with 256 bins on GPU"
# Answer: Use local atomics per work-group, then global atomic merge.
# ===================================================================

HISTOGRAM_KERNEL = """
__kernel void histogram(__global const int* data,
                        __global int* hist,
                        __local int* local_hist,
                        const int N,
                        const int num_bins) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    // Initialize local histogram to zero
    for (int i = lid; i < num_bins; i += lsize) {
        local_hist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each work-item counts into LOCAL histogram (fast)
    if (gid < N) {
        int bin = data[gid];
        atomic_add(&local_hist[bin], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Merge local histogram into GLOBAL histogram
    for (int i = lid; i < num_bins; i += lsize) {
        if (local_hist[i] > 0) {
            atomic_add(&hist[i], local_hist[i]);
        }
    }
}
"""

def demo_histogram(ctx, queue):
    """Demonstrate histogram with local + global atomics."""
    print("\n" + "=" * 70)
    print("  PATTERN 6: HISTOGRAM (Local + Global Atomics)")
    print("  Multiple work-items → same bin requires atomic operations")
    print("=" * 70)

    N = 1 << 20
    NUM_BINS = 256
    LOCAL_SIZE = 256

    data = np.random.randint(0, NUM_BINS, size=N).astype(np.int32)
    hist_gpu = np.zeros(NUM_BINS, dtype=np.int32)

    mf = cl.mem_flags
    d_data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_hist = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hist_gpu)

    prg = cl.Program(ctx, HISTOGRAM_KERNEL).build()
    kernel = cl.Kernel(prg, "histogram")

    global_size = ((N + LOCAL_SIZE - 1) // LOCAL_SIZE) * LOCAL_SIZE
    kernel.set_args(d_data, d_hist, cl.LocalMemory(NUM_BINS * 4),
                    np.int32(N), np.int32(NUM_BINS))
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (LOCAL_SIZE,))
    evt.wait()
    gpu_time_ns = evt.profile.end - evt.profile.start

    cl.enqueue_copy(queue, hist_gpu, d_hist).wait()

    # CPU reference
    t0 = time.perf_counter()
    hist_cpu = np.bincount(data, minlength=NUM_BINS)
    cpu_time = time.perf_counter() - t0

    match = np.array_equal(hist_gpu, hist_cpu)
    print(f"\n  Elements:    {N:,}")
    print(f"  Bins:        {NUM_BINS}")
    print(f"  GPU kernel:  {gpu_time_ns / 1e6:.2f} ms")
    print(f"  CPU (NumPy): {cpu_time * 1000:.2f} ms")
    print(f"  Match:       {'PASS' if match else 'FAIL'}")
    print(f"  Sample bins: GPU={hist_gpu[:5]}  CPU={hist_cpu[:5]}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • atomic_add() — thread-safe increment, serializes conflicting writes
  • Two-level approach: local atomics (fast LDS) → global atomics (slow)
  • Reduces global atomic contention by factor of work-group-size
  • Interview tip: Explain WHY local histogram reduces contention
  • Alternative: privatized histograms (one per work-item, merge later)
""")


# ===================================================================
# PATTERN 7: MATRIX TRANSPOSE (Coalesced Access)
# ===================================================================
# Naive transpose has terrible memory access patterns.
# Tiled transpose uses local memory to convert scattered writes
# into coalesced writes.
#
# Interview Q: "Why is naive transpose slow on GPU? How to fix?"
# Answer: Memory coalescing — adjacent threads must access adjacent
#         addresses. Tile through local memory to fix write pattern.
# ===================================================================

TRANSPOSE_NAIVE_KERNEL = """
__kernel void transpose_naive(__global const float* input,
                              __global float* output,
                              const int rows, const int cols) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
        // Read: coalesced (row-major)
        // Write: STRIDED (col-major) — BAD!
    }
}
"""

TRANSPOSE_TILED_KERNEL = """
#define TILE 16

__kernel void transpose_tiled(__global const float* input,
                              __global float* output,
                              __local float tile[TILE][TILE + 1],  // +1 avoids bank conflicts!
                              const int rows, const int cols) {
    int gx = get_group_id(1) * TILE + get_local_id(0);
    int gy = get_group_id(0) * TILE + get_local_id(1);

    // Load tile from input (coalesced read)
    if (gx < cols && gy < rows) {
        tile[get_local_id(1)][get_local_id(0)] = input[gy * cols + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write transposed tile (coalesced write!)
    gx = get_group_id(0) * TILE + get_local_id(0);
    gy = get_group_id(1) * TILE + get_local_id(1);
    if (gx < rows && gy < cols) {
        output[gy * rows + gx] = tile[get_local_id(0)][get_local_id(1)];
    }
}
"""

def demo_transpose(ctx, queue):
    """Demonstrate naive vs tiled transpose."""
    print("\n" + "=" * 70)
    print("  PATTERN 7: MATRIX TRANSPOSE (Coalesced Access)")
    print("  Tiled transpose fixes strided writes via local memory")
    print("=" * 70)

    ROWS, COLS = 1024, 1024
    TILE = 16
    A = np.random.randn(ROWS, COLS).astype(np.float32)

    mf = cl.mem_flags
    d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    d_out_naive = cl.Buffer(ctx, mf.WRITE_ONLY, A.nbytes)
    d_out_tiled = cl.Buffer(ctx, mf.WRITE_ONLY, A.nbytes)

    prg = cl.Program(ctx, TRANSPOSE_NAIVE_KERNEL + TRANSPOSE_TILED_KERNEL).build()
    k_naive = cl.Kernel(prg, "transpose_naive")
    k_tiled = cl.Kernel(prg, "transpose_tiled")

    gs = (((ROWS + TILE - 1) // TILE) * TILE, ((COLS + TILE - 1) // TILE) * TILE)

    # Naive
    k_naive.set_args(d_A, d_out_naive, np.int32(ROWS), np.int32(COLS))
    for _ in range(3):
        cl.enqueue_nd_range_kernel(queue, k_naive, gs, (TILE, TILE)).wait()
    evt_n = cl.enqueue_nd_range_kernel(queue, k_naive, gs, (TILE, TILE))
    evt_n.wait()
    naive_ms = (evt_n.profile.end - evt_n.profile.start) / 1e6

    # Tiled
    k_tiled.set_args(d_A, d_out_tiled, cl.LocalMemory(TILE * (TILE + 1) * 4),
                     np.int32(ROWS), np.int32(COLS))
    for _ in range(3):
        cl.enqueue_nd_range_kernel(queue, k_tiled, gs, (TILE, TILE)).wait()
    evt_t = cl.enqueue_nd_range_kernel(queue, k_tiled, gs, (TILE, TILE))
    evt_t.wait()
    tiled_ms = (evt_t.profile.end - evt_t.profile.start) / 1e6

    # Verify
    out_naive = np.empty((COLS, ROWS), dtype=np.float32)
    out_tiled = np.empty((COLS, ROWS), dtype=np.float32)
    cl.enqueue_copy(queue, out_naive, d_out_naive).wait()
    cl.enqueue_copy(queue, out_tiled, d_out_tiled).wait()

    ref = A.T
    err_naive = np.max(np.abs(out_naive - ref))
    err_tiled = np.max(np.abs(out_tiled - ref))

    bw_naive = 2 * A.nbytes / (naive_ms / 1000) / 1e9
    bw_tiled = 2 * A.nbytes / (tiled_ms / 1000) / 1e9

    print(f"\n  Matrix:        {ROWS}x{COLS}")
    print(f"  Naive:         {naive_ms:.3f} ms  ({bw_naive:.1f} GB/s)  err={err_naive:.1e}")
    print(f"  Tiled (LDS):   {tiled_ms:.3f} ms  ({bw_tiled:.1f} GB/s)  err={err_tiled:.1e}")
    print(f"  Speedup:       {naive_ms / tiled_ms:.2f}x")
    print(f"  Result:        {'PASS' if err_tiled < 1e-6 else 'FAIL'}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • Memory coalescing: adjacent threads → adjacent addresses = fast
  • Naive transpose: reads coalesced, writes STRIDED = slow
  • Tiled: read tile → local mem → write tile transposed = both coalesced
  • TILE+1 padding prevents local memory bank conflicts
  • Interview tip: Draw the memory access pattern diagram
  • Bandwidth is the metric, not GFLOPS (transpose is memory-bound)
""")


# ===================================================================
# PATTERN 8: SCATTER / GATHER (Indirect Addressing)
# ===================================================================
# Gather: output[i] = input[index[i]]  (indexed read)
# Scatter: output[index[i]] = input[i]  (indexed write)
#
# Interview Q: "Reorder array elements by a permutation on GPU"
# Answer: Gather is preferred (parallel reads are safe).
#         Scatter may have conflicts (two items → same location).
# ===================================================================

GATHER_KERNEL = """
__kernel void gather(__global const float* input,
                     __global const int* indices,
                     __global float* output,
                     const int N) {
    int gid = get_global_id(0);
    if (gid < N) {
        output[gid] = input[indices[gid]];  // Gather: read from scattered locations
    }
}
"""

def demo_scatter_gather(ctx, queue):
    """Demonstrate gather pattern for array permutation."""
    print("\n" + "=" * 70)
    print("  PATTERN 8: SCATTER / GATHER (Indirect Addressing)")
    print("  Gather: safe parallel reads from scattered locations")
    print("=" * 70)

    N = 1 << 20
    data = np.random.randn(N).astype(np.float32)
    indices = np.random.permutation(N).astype(np.int32)

    mf = cl.mem_flags
    d_data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_idx = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indices)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    prg = cl.Program(ctx, GATHER_KERNEL).build()
    kernel = cl.Kernel(prg, "gather")
    kernel.set_args(d_data, d_idx, d_out, np.int32(N))

    evt = cl.enqueue_nd_range_kernel(queue, kernel, (N,), None)
    evt.wait()
    gpu_time_ns = evt.profile.end - evt.profile.start

    result = np.empty_like(data)
    cl.enqueue_copy(queue, result, d_out).wait()

    t0 = time.perf_counter()
    cpu_result = data[indices]
    cpu_time = time.perf_counter() - t0

    match = np.allclose(result, cpu_result)
    print(f"\n  Elements:    {N:,}")
    print(f"  GPU kernel:  {gpu_time_ns / 1e6:.2f} ms")
    print(f"  CPU (NumPy): {cpu_time * 1000:.2f} ms")
    print(f"  Result:      {'PASS' if match else 'FAIL'}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • Gather = read from scattered addresses → safe (no write conflicts)
  • Scatter = write to scattered addresses → may need atomics
  • Gather is always preferred when possible (rewrite scatter as gather)
  • Cache performance: random access patterns → poor cache utilization
  • Interview tip: "Invert the permutation to convert scatter to gather"
""")


# ===================================================================
# PATTERN 9: DOT PRODUCT (Map + Reduce Fusion)
# ===================================================================
# Fusing map and reduce is a common optimization.
#   dot(A, B) = sum(A[i] * B[i])  =  map(multiply) + reduce(sum)
#
# Interview Q: "Compute dot product of two large vectors on GPU"
# Answer: Fuse multiply into the reduce kernel — saves a pass.
# ===================================================================

DOT_KERNEL = """
__kernel void dot_product(__global const float* A,
                          __global const float* B,
                          __global float* partial_sums,
                          __local float* scratch,
                          const int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // FUSED: multiply + load into local memory
    scratch[lid] = (gid < N) ? A[gid] * B[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduce
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_sums[get_group_id(0)] = scratch[0];
    }
}
"""

def demo_dot_product(ctx, queue):
    """Demonstrate fused map+reduce for dot product."""
    print("\n" + "=" * 70)
    print("  PATTERN 9: DOT PRODUCT (Fused Map + Reduce)")
    print("  Multiply + sum in one kernel — avoids intermediate buffer")
    print("=" * 70)

    N = 1 << 20
    LOCAL_SIZE = 256
    A = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)

    num_groups = (N + LOCAL_SIZE - 1) // LOCAL_SIZE
    global_size = num_groups * LOCAL_SIZE

    mf = cl.mem_flags
    d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    d_partial = cl.Buffer(ctx, mf.WRITE_ONLY, num_groups * 4)

    prg = cl.Program(ctx, DOT_KERNEL).build()
    kernel = cl.Kernel(prg, "dot_product")
    kernel.set_args(d_A, d_B, d_partial, cl.LocalMemory(LOCAL_SIZE * 4), np.int32(N))

    evt = cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (LOCAL_SIZE,))
    evt.wait()
    gpu_time_ns = evt.profile.end - evt.profile.start

    partial = np.empty(num_groups, dtype=np.float32)
    cl.enqueue_copy(queue, partial, d_partial).wait()
    gpu_dot = np.sum(partial)

    cpu_dot = np.dot(A, B)
    rel_err = abs(gpu_dot - cpu_dot) / max(abs(cpu_dot), 1e-10)

    print(f"\n  Vector size: {N:,}")
    print(f"  GPU dot:     {gpu_dot:.4f}")
    print(f"  CPU dot:     {cpu_dot:.4f}")
    print(f"  Rel error:   {rel_err:.2e}")
    print(f"  GPU kernel:  {gpu_time_ns / 1e6:.2f} ms")
    print(f"  Result:      {'PASS' if rel_err < 1e-3 else 'FAIL'}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • Kernel fusion: combine map + reduce to save memory bandwidth
  • One less global memory pass = significant speedup
  • Same tree-reduce pattern from Session 1
  • Interview tip: Always look for fusion opportunities
  • Extends to: cosine similarity, L2 norm, softmax
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 2: ADVANCED GPU PATTERNS")
    print("  Histogram, Transpose, Scatter/Gather, Dot Product")
    print("=" * 70)

    ctx, queue, device = get_gpu_context()

    demo_histogram(ctx, queue)
    demo_transpose(ctx, queue)
    demo_scatter_gather(ctx, queue)
    demo_dot_product(ctx, queue)

    print("\n" + "=" * 70)
    print("  SESSION 2 COMPLETE — SUMMARY")
    print("=" * 70)
    print("""
  Patterns learned:
  ─────────────────
  6. Histogram     — Two-level atomics (local → global)
  7. Transpose     — Tiled with LDS to fix coalescing
  8. Scatter/Gather — Gather is safe; convert scatter to gather
  9. Dot Product   — Fused map+reduce for fewer memory passes

  Memory access rules of thumb:
  ─────────────────────────────
  • Coalesced access: adjacent threads → adjacent addresses
  • Bank conflicts: threads hitting same LDS bank → serialized
  • TILE+1 padding trick avoids bank conflicts in transpose
  • Atomic operations: use local first, merge to global

  Next: Session 3 — NPU inference patterns (quantization, ONNX, VitisAI)
""")


if __name__ == "__main__":
    main()
