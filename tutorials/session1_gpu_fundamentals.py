"""
Session 1: GPU Parallel Fundamentals — OpenCL on AMD Radeon 860M
=================================================================
Learn the foundational parallel patterns that every GPU programmer must know.

Topics:
  1. Work-items & Work-groups (GPU execution model)
  2. Pattern: Parallel Map (element-wise transform)
  3. Pattern: Parallel Reduce (sum, max, dot product)
  4. Pattern: Prefix Sum / Scan (inclusive & exclusive)
  5. Memory hierarchy: Global vs Local vs Private

Hardware: AMD Radeon 860M (RDNA 3.5, 4 CUs, OpenCL 2.0)
Run: conda run -n ryzen-ai-1.7.1 python tutorials/session1_gpu_fundamentals.py
"""

import numpy as np
import pyopencl as cl
import time

# ---------------------------------------------------------------------------
# Setup: Get the AMD GPU
# ---------------------------------------------------------------------------

def get_gpu_context():
    """Create an OpenCL context on the AMD GPU."""
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if device.type == cl.device_type.GPU:
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                print(f"  GPU: {device.name}")
                print(f"  Compute Units: {device.max_compute_units}")
                print(f"  Max Work Group Size: {device.max_work_group_size}")
                print(f"  Local Memory: {device.local_mem_size // 1024} KB")
                return ctx, queue, device
    raise RuntimeError("No GPU found")


# ===================================================================
# PATTERN 1: PARALLEL MAP
# ===================================================================
# Map applies the SAME function to EVERY element independently.
# This is the simplest and most common GPU pattern.
#
#   Input:  [a, b, c, d, e, ...]
#   Output: [f(a), f(b), f(c), f(d), f(e), ...]
#
# Interview Q: "How would you apply a transformation to 10M elements?"
# Answer: Parallel Map — one work-item per element, no synchronization.
# ===================================================================

MAP_KERNEL = """
// Each work-item processes ONE element
__kernel void vector_map(__global const float* input,
                         __global float* output,
                         const int N) {
    int gid = get_global_id(0);  // Unique ID for this work-item
    if (gid < N) {
        // Apply: f(x) = x^2 + 2x + 1 = (x+1)^2
        float x = input[gid];
        output[gid] = x * x + 2.0f * x + 1.0f;
    }
}
"""

def demo_parallel_map(ctx, queue):
    """Demonstrate the Parallel Map pattern."""
    print("\n" + "=" * 70)
    print("  PATTERN 1: PARALLEL MAP")
    print("  One work-item per element, no synchronization needed")
    print("=" * 70)

    N = 1 << 20  # 1M elements
    data = np.random.randn(N).astype(np.float32)

    mf = cl.mem_flags
    d_input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_output = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    prg = cl.Program(ctx, MAP_KERNEL).build()
    kernel = cl.Kernel(prg, "vector_map")

    # Run on GPU
    kernel.set_args(d_input, d_output, np.int32(N))
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (N,), None)
    evt.wait()
    gpu_time_ns = evt.profile.end - evt.profile.start

    gpu_result = np.empty_like(data)
    cl.enqueue_copy(queue, gpu_result, d_output).wait()

    # Run on CPU for comparison
    t0 = time.perf_counter()
    cpu_result = data ** 2 + 2 * data + 1
    cpu_time = time.perf_counter() - t0

    # Verify
    max_err = np.max(np.abs(gpu_result - cpu_result))
    print(f"\n  Elements:    {N:,}")
    print(f"  GPU kernel:  {gpu_time_ns / 1e6:.2f} ms")
    print(f"  CPU (NumPy): {cpu_time * 1000:.2f} ms")
    print(f"  Max error:   {max_err:.2e}")
    print(f"  Speedup:     {cpu_time / (gpu_time_ns / 1e9):.1f}x")
    print(f"  Result:      {'PASS' if max_err < 1e-4 else 'FAIL'}")

    print("""
  KEY CONCEPT:
  ─────────────
  • get_global_id(0) gives each work-item a unique index
  • No barriers or synchronization needed — elements are independent
  • Perfect for: color transforms, activation functions, normalization
  • Interview tip: Map is "embarrassingly parallel" — linear speedup
""")


# ===================================================================
# PATTERN 2: PARALLEL REDUCE
# ===================================================================
# Reduce combines all elements into a single value using an associative
# operator (sum, max, min, product).
#
#   Input:  [a, b, c, d, e, f, g, h]
#   Step 1: [a+b, c+d, e+f, g+h]
#   Step 2: [(a+b)+(c+d), (e+f)+(g+h)]
#   Step 3: [((a+b)+(c+d))+((e+f)+(g+h))]
#
# Interview Q: "Find the sum of 10M numbers on a GPU"
# Answer: Tree-based parallel reduction in shared/local memory.
# ===================================================================

REDUCE_KERNEL = """
__kernel void parallel_reduce(__global const float* input,
                              __global float* partial_sums,
                              __local float* scratch,
                              const int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);          // Index within work-group
    int group_size = get_local_size(0);  // Work-group size

    // Step 1: Each work-item loads one element into local memory
    scratch[lid] = (gid < N) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);  // Wait for all loads

    // Step 2: Tree reduction in local memory
    //   Stride: group_size/2, group_size/4, ..., 1
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Sync after each level
    }

    // Step 3: Work-item 0 writes the partial sum for this group
    if (lid == 0) {
        partial_sums[get_group_id(0)] = scratch[0];
    }
}
"""

def demo_parallel_reduce(ctx, queue):
    """Demonstrate the Parallel Reduce pattern."""
    print("\n" + "=" * 70)
    print("  PATTERN 2: PARALLEL REDUCE (Tree Reduction)")
    print("  Combine N elements → 1 value using local memory")
    print("=" * 70)

    N = 1 << 20  # 1M elements
    LOCAL_SIZE = 256
    data = np.random.randn(N).astype(np.float32)

    # Number of work-groups
    num_groups = (N + LOCAL_SIZE - 1) // LOCAL_SIZE

    mf = cl.mem_flags
    d_input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_partial = cl.Buffer(ctx, mf.WRITE_ONLY, num_groups * 4)

    prg = cl.Program(ctx, REDUCE_KERNEL).build()
    kernel = cl.Kernel(prg, "parallel_reduce")

    kernel.set_args(d_input, d_partial, cl.LocalMemory(LOCAL_SIZE * 4), np.int32(N))
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (num_groups * LOCAL_SIZE,), (LOCAL_SIZE,))
    evt.wait()
    gpu_time_ns = evt.profile.end - evt.profile.start

    # Read partial sums and finish on CPU
    partial = np.empty(num_groups, dtype=np.float32)
    cl.enqueue_copy(queue, partial, d_partial).wait()
    gpu_sum = np.sum(partial)  # Final reduction of partial sums

    # CPU reference
    t0 = time.perf_counter()
    cpu_sum = np.sum(data)
    cpu_time = time.perf_counter() - t0

    rel_err = abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-10)
    print(f"\n  Elements:    {N:,}")
    print(f"  GPU sum:     {gpu_sum:.4f}")
    print(f"  CPU sum:     {cpu_sum:.4f}")
    print(f"  Rel error:   {rel_err:.2e}")
    print(f"  GPU kernel:  {gpu_time_ns / 1e6:.2f} ms")
    print(f"  CPU (NumPy): {cpu_time * 1000:.2f} ms")
    print(f"  Result:      {'PASS' if rel_err < 1e-3 else 'FAIL'}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • barrier(CLK_LOCAL_MEM_FENCE) synchronizes within a work-group
  • Tree reduction: O(log N) steps instead of O(N) sequential
  • Local memory (__local) is fast scratchpad shared by work-group
  • Two-phase: GPU produces partial sums, CPU/GPU finishes
  • Interview tip: Know the stride pattern and why barriers are needed
""")


# ===================================================================
# PATTERN 3: PREFIX SUM (Exclusive Scan)
# ===================================================================
# Scan computes running totals. Essential building block for:
#   - Stream compaction, radix sort, histogram, load balancing
#
#   Input:    [3, 1, 7, 0, 4, 1, 6, 3]
#   Exc.Scan: [0, 3, 4, 11, 11, 15, 16, 22]  (shifted right, start=0)
#   Inc.Scan: [3, 4, 11, 11, 15, 16, 22, 25]
#
# Interview Q: "Implement parallel prefix sum"
# Answer: Blelloch scan — up-sweep (reduce) then down-sweep (scatter)
# ===================================================================

SCAN_KERNEL = """
// Hillis-Steele inclusive scan (work-group level)
__kernel void inclusive_scan(__global const float* input,
                            __global float* output,
                            __local float* temp,
                            const int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int n = get_local_size(0);

    // Load into local memory
    temp[lid] = (gid < N) ? input[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Hillis-Steele: each step doubles the stride
    // Step d: temp[i] += temp[i - 2^d]
    for (int stride = 1; stride < n; stride <<= 1) {
        float val = 0.0f;
        if (lid >= stride) {
            val = temp[lid - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[lid] += val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < N) {
        output[gid] = temp[lid];
    }
}
"""

def demo_prefix_scan(ctx, queue):
    """Demonstrate the Prefix Sum / Scan pattern."""
    print("\n" + "=" * 70)
    print("  PATTERN 3: PREFIX SUM (Inclusive Scan)")
    print("  Running totals — building block for sort, compact, histogram")
    print("=" * 70)

    # Use a small array to verify correctness visually
    N = 256
    LOCAL_SIZE = 256
    data = np.random.randint(0, 10, size=N).astype(np.float32)

    mf = cl.mem_flags
    d_input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_output = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    prg = cl.Program(ctx, SCAN_KERNEL).build()
    kernel = cl.Kernel(prg, "inclusive_scan")

    kernel.set_args(d_input, d_output, cl.LocalMemory(LOCAL_SIZE * 4), np.int32(N))
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (LOCAL_SIZE,), (LOCAL_SIZE,))
    evt.wait()
    gpu_time_ns = evt.profile.end - evt.profile.start

    gpu_result = np.empty_like(data)
    cl.enqueue_copy(queue, gpu_result, d_output).wait()

    cpu_result = np.cumsum(data)

    max_err = np.max(np.abs(gpu_result - cpu_result))
    print(f"\n  Elements:   {N}")
    print(f"  Input[:8]:  {data[:8].astype(int)}")
    print(f"  GPU[:8]:    {gpu_result[:8].astype(int)}")
    print(f"  CPU[:8]:    {cpu_result[:8].astype(int)}")
    print(f"  GPU kernel: {gpu_time_ns / 1e6:.3f} ms")
    print(f"  Max error:  {max_err:.2e}")
    print(f"  Result:     {'PASS' if max_err < 1e-2 else 'FAIL'}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • Hillis-Steele: O(N log N) work but only O(log N) steps — fast on GPU
  • Blelloch scan: O(N) work, O(log N) steps — more work-efficient
  • Two barriers per step: read-then-write to avoid race conditions
  • Building block for: radix sort, stream compaction, histogram eq.
  • Interview tip: Know BOTH Hillis-Steele and Blelloch, trade-offs
""")


# ===================================================================
# PATTERN 4: STENCIL / CONVOLUTION
# ===================================================================
# Each output element depends on a NEIGHBORHOOD of input elements.
# Common in: image processing, PDEs, neural network convolutions.
#
#   1D example (3-point stencil):
#     output[i] = 0.25*input[i-1] + 0.5*input[i] + 0.25*input[i+1]
#
# Interview Q: "How do you handle boundary data in GPU stencils?"
# Answer: Halo/ghost cells in local memory, barrier before compute.
# ===================================================================

STENCIL_KERNEL = """
// 1D smoothing stencil with local memory + halo cells
__kernel void stencil_1d(__global const float* input,
                         __global float* output,
                         __local float* tile,
                         const int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int HALO = 1;  // stencil radius

    // Load center data into local memory
    int tile_idx = lid + HALO;
    tile[tile_idx] = (gid < N) ? input[gid] : 0.0f;

    // Load halo (ghost) cells at the edges of the work-group
    if (lid < HALO) {
        int left_idx = gid - HALO;
        tile[lid] = (left_idx >= 0) ? input[left_idx] : 0.0f;
    }
    if (lid >= lsize - HALO) {
        int right_idx = gid + HALO;
        tile[tile_idx + HALO] = (right_idx < N) ? input[right_idx] : 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Apply stencil: weighted average of neighbors
    if (gid < N) {
        output[gid] = 0.25f * tile[tile_idx - 1]
                    + 0.50f * tile[tile_idx]
                    + 0.25f * tile[tile_idx + 1];
    }
}
"""

def demo_stencil(ctx, queue):
    """Demonstrate the Stencil pattern with halo cells."""
    print("\n" + "=" * 70)
    print("  PATTERN 4: STENCIL (1D Smoothing with Halo Cells)")
    print("  Each element depends on its neighbors — needs local mem + halo")
    print("=" * 70)

    N = 1 << 20
    LOCAL_SIZE = 256
    HALO = 1
    data = np.random.randn(N).astype(np.float32)

    mf = cl.mem_flags
    d_input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_output = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    prg = cl.Program(ctx, STENCIL_KERNEL).build()
    kernel = cl.Kernel(prg, "stencil_1d")

    local_mem_size = (LOCAL_SIZE + 2 * HALO) * 4
    kernel.set_args(d_input, d_output, cl.LocalMemory(local_mem_size), np.int32(N))

    global_size = ((N + LOCAL_SIZE - 1) // LOCAL_SIZE) * LOCAL_SIZE
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (LOCAL_SIZE,))
    evt.wait()
    gpu_time_ns = evt.profile.end - evt.profile.start

    gpu_result = np.empty_like(data)
    cl.enqueue_copy(queue, gpu_result, d_output).wait()

    # CPU reference
    t0 = time.perf_counter()
    cpu_result = np.copy(data)
    cpu_result[1:-1] = 0.25 * data[:-2] + 0.5 * data[1:-1] + 0.25 * data[2:]
    cpu_result[0] = 0.5 * data[0] + 0.25 * data[1]
    cpu_result[-1] = 0.25 * data[-2] + 0.5 * data[-1]
    cpu_time = time.perf_counter() - t0

    # Compare interior (skip boundaries)
    max_err = np.max(np.abs(gpu_result[1:-1] - cpu_result[1:-1]))
    print(f"\n  Elements:    {N:,}")
    print(f"  GPU kernel:  {gpu_time_ns / 1e6:.2f} ms")
    print(f"  CPU (NumPy): {cpu_time * 1000:.2f} ms")
    print(f"  Max error:   {max_err:.2e}")
    print(f"  Result:      {'PASS' if max_err < 1e-5 else 'FAIL'}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • Halo/ghost cells: load extra boundary data for neighbor access
  • Local memory tile = work-group data + halo on each side
  • barrier() ensures halos are loaded before stencil computation
  • Generalizes to 2D/3D (image filters, PDE solvers, convolutions)
  • Interview tip: Explain the halo region and why it avoids global reads
""")


# ===================================================================
# PATTERN 5: MATRIX MULTIPLY (Tiled)
# ===================================================================
# The "Hello World" of GPU computing. Key interview pattern.
# Tiling uses local memory to reduce global memory traffic.
#
# Interview Q: "Optimize matrix multiply on a GPU"
# Answer: Tile into work-group-sized blocks, load tiles into shared
#         memory, compute partial products, accumulate.
# ===================================================================

MATMUL_TILED_KERNEL = """
#define TILE 16

__kernel void matmul_tiled(__global const float* A,
                           __global const float* B,
                           __global float* C,
                           const int M, const int K, const int N) {
    __local float tileA[TILE][TILE];
    __local float tileB[TILE][TILE];

    int row = get_local_id(0);
    int col = get_local_id(1);
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);

    float sum = 0.0f;

    // Iterate over tiles along the K dimension
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        // Collaborative load: each work-item loads one element per tile
        int aCol = t * TILE + col;
        int bRow = t * TILE + row;

        tileA[row][col] = (globalRow < M && aCol < K) ? A[globalRow * K + aCol] : 0.0f;
        tileB[row][col] = (bRow < K && globalCol < N) ? B[bRow * N + globalCol] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the tile
        for (int k = 0; k < TILE; k++) {
            sum += tileA[row][k] * tileB[k][col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = sum;
    }
}
"""

def demo_matmul_tiled(ctx, queue):
    """Demonstrate tiled matrix multiply."""
    print("\n" + "=" * 70)
    print("  PATTERN 5: TILED MATRIX MULTIPLY")
    print("  Collaborative loading into local memory reduces global traffic")
    print("=" * 70)

    M, K, N = 512, 512, 512
    TILE = 16
    A = np.random.randn(M, K).astype(np.float32) * 0.01
    B = np.random.randn(K, N).astype(np.float32) * 0.01

    mf = cl.mem_flags
    d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    d_C = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)

    prg = cl.Program(ctx, MATMUL_TILED_KERNEL).build()
    kernel = cl.Kernel(prg, "matmul_tiled")

    kernel.set_args(d_A, d_B, d_C, np.int32(M), np.int32(K), np.int32(N))

    global_size = (((M + TILE - 1) // TILE) * TILE, ((N + TILE - 1) // TILE) * TILE)

    # Warmup
    for _ in range(3):
        cl.enqueue_nd_range_kernel(queue, kernel, global_size, (TILE, TILE)).wait()

    # Timed
    times = []
    for _ in range(10):
        evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, (TILE, TILE))
        evt.wait()
        times.append((evt.profile.end - evt.profile.start) / 1e6)

    C_gpu = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, C_gpu, d_C).wait()

    C_cpu = A @ B
    max_err = np.max(np.abs(C_gpu - C_cpu))
    avg_ms = np.mean(times)
    gflops = (2 * M * K * N) / (avg_ms * 1e6)

    print(f"\n  Matrix:      {M}x{K} @ {K}x{N}")
    print(f"  Tile size:   {TILE}x{TILE}")
    print(f"  GPU kernel:  {avg_ms:.2f} ms")
    print(f"  Throughput:  {gflops:.1f} GFLOPS")
    print(f"  Max error:   {max_err:.2e}")
    print(f"  Result:      {'PASS' if max_err < 0.1 else 'FAIL'}")

    print("""
  KEY CONCEPTS:
  ─────────────
  • Tiling: break matrix into TILE×TILE blocks loaded into __local
  • Each tile loaded collaboratively: 1 element per work-item
  • Reduces global memory reads from O(N³) to O(N³/TILE)
  • Two barriers per tile: after load, after compute
  • Interview tip: Explain WHY tiling helps (memory bandwidth is the bottleneck)
  • Follow-up: register tiling, vectorized loads, double buffering
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 1: GPU PARALLEL FUNDAMENTALS")
    print("  AMD Radeon 860M (RDNA 3.5) — OpenCL 2.0")
    print("=" * 70)

    ctx, queue, device = get_gpu_context()

    demo_parallel_map(ctx, queue)
    demo_parallel_reduce(ctx, queue)
    demo_prefix_scan(ctx, queue)
    demo_stencil(ctx, queue)
    demo_matmul_tiled(ctx, queue)

    print("\n" + "=" * 70)
    print("  SESSION 1 COMPLETE — SUMMARY")
    print("=" * 70)
    print("""
  Patterns learned:
  ─────────────────
  1. Map       — Independent per-element transform (embarrassingly parallel)
  2. Reduce    — Tree reduction in local memory with barriers
  3. Scan      — Prefix sum (Hillis-Steele), building block for sort
  4. Stencil   — Neighbor access with halo/ghost cells in local memory
  5. MatMul    — Tiled computation to exploit memory hierarchy

  Memory hierarchy (fastest → slowest):
  ──────────────────────────────────────
  Private registers  >  __local (LDS)  >  __global (VRAM/DDR)

  Next: Session 2 — Advanced GPU patterns (histogram, transpose, sort)
""")


if __name__ == "__main__":
    main()
