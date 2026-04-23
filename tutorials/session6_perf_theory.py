"""
Session 6: Performance Theory & Scaling Laws
=============================================
The theory interviewers ALWAYS ask about. With live measurements.

Topics:
  1. Amdahl's Law — speedup limited by serial fraction
  2. Gustafson's Law — scaled speedup for larger problems
  3. Roofline Model — compute-bound vs memory-bound classification
  4. Double Buffering — overlap transfer with compute (implemented)
  5. Kernel Fusion — reduce memory passes for better performance

Hardware: AMD Radeon 860M (RDNA 3.5) + Zen 5 CPU
Run: conda run -n ryzen-ai-1.7.1 --no-capture-output python tutorials/session6_perf_theory.py
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
                return ctx, queue, device
    raise RuntimeError("No GPU found")


def section(num, title, subtitle=""):
    print(f"\n{'=' * 70}")
    print(f"  {num}. {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70)


# ===================================================================
# 1. AMDAHL'S LAW
# ===================================================================
# "What limits your parallel speedup?"
#
#   Speedup(P) = 1 / (S + (1-S)/P)
#
#   S = serial fraction (0 to 1)
#   P = number of processors
#
# Even with infinite processors, speedup <= 1/S
# ===================================================================

def demo_amdahls_law(ctx, queue, device):
    section("1", "AMDAHL'S LAW",
            "Speedup = 1 / (S + (1-S)/P)  where S = serial fraction")

    print("""
  Formula:   Speedup(P) = 1 / ( S + (1-S)/P )

  S = fraction of work that MUST be serial
  P = number of parallel processors
  As P -> infinity:  Speedup_max = 1/S

  Example: if 10% is serial (S=0.1), max speedup = 10x
           even with 1000 processors!

  ┌──────────────────────────────────────────────────────────┐
  │  Serial     Speedup   Speedup   Speedup     Maximum     │
  │  Fraction   P=4       P=16      P=64        (P=inf)     │
  │──────────────────────────────────────────────────────────│""")

    for s in [0.01, 0.05, 0.10, 0.25, 0.50]:
        su4 = 1.0 / (s + (1-s)/4)
        su16 = 1.0 / (s + (1-s)/16)
        su64 = 1.0 / (s + (1-s)/64)
        su_max = 1.0 / s
        print(f"  │  S={s:.2f}     {su4:5.2f}x    {su16:5.2f}x    "
              f"{su64:5.2f}x      {su_max:6.1f}x     │")

    print("  └──────────────────────────────────────────────────────────┘")

    # Live demonstration with actual serial + parallel sections
    N = 1 << 22  # 4M elements
    data = np.random.randn(N).astype(np.float32)

    KERNEL = """
    __kernel void compute(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) {
            float x = in[i];
            // Do some work to make it measurable
            for (int k = 0; k < 10; k++)
                x = x * 1.001f + 0.001f;
            out[i] = x;
        }
    }
    """
    prg = cl.Program(ctx, KERNEL).build()
    mf = cl.mem_flags

    # Simulate different serial fractions
    print(f"\n  Live demo: {N:,} elements, varying serial fraction\n")
    print(f"  {'Serial Work':>14s} {'Serial(ms)':>11s} {'GPU(ms)':>9s} "
          f"{'Total(ms)':>10s} {'Speedup':>8s} {'Predicted':>10s}")
    print("  " + "-" * 68)

    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    # Get baseline GPU time
    evt = prg.compute(queue, (N,), (256,), d_in, d_out, np.int32(N))
    evt.wait()
    # Warmup done, now measure
    gpu_times = []
    for _ in range(5):
        evt = prg.compute(queue, (N,), (256,), d_in, d_out, np.int32(N))
        evt.wait()
        gpu_times.append((evt.profile.end - evt.profile.start) / 1e6)
    gpu_ms = np.mean(gpu_times)

    # Measure CPU-only (serial baseline)
    t0 = time.perf_counter()
    result_cpu = data.copy()
    for _ in range(10):
        result_cpu = result_cpu * 1.001 + 0.001
    cpu_total_ms = (time.perf_counter() - t0) * 1000

    for serial_pct in [1, 5, 10, 25, 50]:
        # Simulate serial work as fraction of CPU total
        serial_ms = cpu_total_ms * (serial_pct / 100.0)
        # Actual total = serial + GPU parallel
        total_ms = serial_ms + gpu_ms
        # Pure CPU baseline for this workload
        cpu_baseline = cpu_total_ms
        actual_speedup = cpu_baseline / total_ms
        # Amdahl prediction (P = effective parallelism)
        S = serial_pct / 100.0
        P = cpu_total_ms / gpu_ms  # effective parallelism from GPU
        predicted = 1.0 / (S + (1-S)/P)
        print(f"  {serial_pct:12d}% {serial_ms:11.2f} {gpu_ms:9.2f} "
              f"{total_ms:10.2f} {actual_speedup:7.2f}x {predicted:9.2f}x")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • The serial fraction DOMINATES — reducing it matters more than adding cores
  • Profile first to find the serial bottleneck
  • Common serial parts: data loading, memory allocation, result collection
  • Amdahl's Law is PESSIMISTIC — assumes fixed problem size
  • Follow-up: "What about Gustafson's Law?" → see next section
""")


# ===================================================================
# 2. GUSTAFSON'S LAW
# ===================================================================
# "What if we scale the problem with the processors?"
#
#   Scaled Speedup(P) = P - S*(P-1)
#
#   Instead of fixed work, we do MORE work with more processors.
#   This is what real GPU programs do — bigger data, not same data faster.
# ===================================================================

def demo_gustafsons_law(ctx, queue, device):
    section("2", "GUSTAFSON'S LAW",
            "Scaled Speedup = P - S*(P-1)  — scale the problem, not just speed")

    print("""
  Amdahl:    "I have a FIXED problem. How fast with P processors?"
  Gustafson: "I have P processors. How much MORE can I do?"

  Formula:   Scaled_Speedup(P) = P - S*(P-1)

  This is OPTIMISTIC — as problem grows, serial fraction shrinks!

  ┌──────────────────────────────────────────────────────────┐
  │  Serial     Amdahl      Gustafson   Gustafson   Gustafson│
  │  Fraction   P=64        P=64        P=256       P=1024   │
  │──────────────────────────────────────────────────────────│""")

    for s in [0.01, 0.05, 0.10, 0.25]:
        amdahl = 1.0 / (s + (1-s)/64)
        gust64 = 64 - s * (64 - 1)
        gust256 = 256 - s * (256 - 1)
        gust1024 = 1024 - s * (1024 - 1)
        print(f"  │  S={s:.2f}     {amdahl:5.1f}x      {gust64:5.1f}x      "
              f"{gust256:6.1f}x     {gust1024:7.1f}x  │")

    print("  └──────────────────────────────────────────────────────────┘")

    # Live demo: process increasingly larger data on GPU
    mf = cl.mem_flags
    KERNEL = """
    __kernel void saxpy(__global const float* X, __global float* Y,
                        float a, int N) {
        int i = get_global_id(0);
        if (i < N) Y[i] = a * X[i] + Y[i];
    }
    """
    prg = cl.Program(ctx, KERNEL).build()

    print(f"\n  Live demo: scale problem with GPU — Gustafson in action\n")
    print(f"  {'N':>12s} {'CPU(ms)':>9s} {'GPU(ms)':>9s} "
          f"{'Speedup':>8s} {'Efficiency':>11s}")
    print("  " + "-" * 55)

    for exp in range(16, 25):
        N = 1 << exp
        X = np.random.randn(N).astype(np.float32)
        Y = np.random.randn(N).astype(np.float32)

        # CPU
        t0 = time.perf_counter()
        for _ in range(5):
            _ = 2.0 * X + Y
        cpu_ms = (time.perf_counter() - t0) / 5 * 1000

        # GPU
        d_X = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
        d_Y = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=Y)
        # Warmup
        prg.saxpy(queue, (N,), (256,), d_X, d_Y, np.float32(2.0), np.int32(N)).wait()
        gpu_times = []
        for _ in range(5):
            cl.enqueue_copy(queue, d_Y, Y)  # Reset
            evt = prg.saxpy(queue, (N,), (256,), d_X, d_Y,
                            np.float32(2.0), np.int32(N))
            evt.wait()
            gpu_times.append((evt.profile.end - evt.profile.start) / 1e6)
        gpu_ms = np.mean(gpu_times)

        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
        # Efficiency = speedup / theoretical_peak
        cus = device.max_compute_units
        efficiency = speedup / cus * 100 if cus > 0 else 0

        print(f"  {N:>12,d} {cpu_ms:>9.3f} {gpu_ms:>9.3f} "
              f"{speedup:>7.1f}x {efficiency:>9.1f}%")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • Gustafson: bigger problems justify more parallelism
  • GPU sweet spot: large data where parallel portion dominates
  • Efficiency = speedup / num_processors — measures utilization
  • Real-world: we rarely solve the SAME problem faster;
    we solve BIGGER problems in the same time
  • "Weak scaling" (Gustafson) vs "Strong scaling" (Amdahl)
""")


# ===================================================================
# 3. ROOFLINE MODEL
# ===================================================================
# Classifies kernels as COMPUTE-BOUND or MEMORY-BOUND.
#
#   Arithmetic Intensity (AI) = FLOPs / Bytes transferred
#   Performance = min(Peak_FLOPS, Peak_BW * AI)
#
# If AI is low → memory-bound (optimize memory access)
# If AI is high → compute-bound (optimize ALU usage)
# ===================================================================

def demo_roofline(ctx, queue, device):
    section("3", "ROOFLINE MODEL",
            "Arithmetic Intensity = FLOPs / Bytes → compute or memory bound?")

    # Measure peak bandwidth
    N = 1 << 22
    mf = cl.mem_flags
    data = np.random.randn(N).astype(np.float32)

    K_COPY = """
    __kernel void copy(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) out[i] = in[i];
    }
    """
    K_LOW_AI = """
    // AI = 1 FLOP / 8 bytes (read + write) = 0.125 FLOP/byte
    __kernel void add(__global const float* A, __global const float* B,
                      __global float* C, int N) {
        int i = get_global_id(0);
        if (i < N) C[i] = A[i] + B[i];
    }
    """
    K_HIGH_AI = """
    // AI = 100 FLOPs / 8 bytes = 12.5 FLOP/byte
    __kernel void heavy(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) {
            float x = in[i];
            for (int k = 0; k < 50; k++)
                x = x * x + 0.5f;  // 2 FLOPs per iteration
            out[i] = x;
        }
    }
    """

    prg_copy = cl.Program(ctx, K_COPY).build()
    prg_low = cl.Program(ctx, K_LOW_AI).build()
    prg_high = cl.Program(ctx, K_HIGH_AI).build()

    d_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    # Measure peak bandwidth via copy
    prg_copy.copy(queue, (N,), (256,), d_A, d_out, np.int32(N)).wait()
    bw_times = []
    for _ in range(10):
        evt = prg_copy.copy(queue, (N,), (256,), d_A, d_out, np.int32(N))
        evt.wait()
        bw_times.append((evt.profile.end - evt.profile.start) / 1e6)
    copy_ms = np.mean(bw_times)
    peak_bw = (N * 4 * 2) / (copy_ms / 1000) / 1e9  # GB/s (read + write)

    # Low AI kernel: vector add
    prg_low.add(queue, (N,), (256,), d_A, d_B, d_out, np.int32(N)).wait()
    low_times = []
    for _ in range(10):
        evt = prg_low.add(queue, (N,), (256,), d_A, d_B, d_out, np.int32(N))
        evt.wait()
        low_times.append((evt.profile.end - evt.profile.start) / 1e6)
    low_ms = np.mean(low_times)
    low_flops = N  # 1 add per element
    low_bytes = N * 4 * 3  # read A, read B, write C
    low_ai = low_flops / low_bytes
    low_gflops = low_flops / (low_ms / 1000) / 1e9

    # High AI kernel: heavy compute
    prg_high.heavy(queue, (N,), (256,), d_A, d_out, np.int32(N)).wait()
    high_times = []
    for _ in range(10):
        evt = prg_high.heavy(queue, (N,), (256,), d_A, d_out, np.int32(N))
        evt.wait()
        high_times.append((evt.profile.end - evt.profile.start) / 1e6)
    high_ms = np.mean(high_times)
    high_flops = N * 100  # 50 iterations * 2 FLOPs
    high_bytes = N * 4 * 2  # read + write
    high_ai = high_flops / high_bytes
    high_gflops = high_flops / (high_ms / 1000) / 1e9

    # Ridge point
    peak_gflops_est = high_gflops * 1.2  # rough estimate
    ridge_point = peak_gflops_est / peak_bw

    print(f"""
  YOUR GPU: {device.name}
  Measured peak bandwidth: {peak_bw:.1f} GB/s (copy kernel)

  ┌────────────────────────────────────────────────────────────┐
  │  ROOFLINE MODEL                                            │
  │                                                            │
  │  Performance                                               │
  │  (GFLOPS)     .........*─────────── Peak Compute           │
  │        ┌─── ridge point                                    │
  │        │  ./                                               │
  │       ./ .                                                 │
  │      / .   memory-bound region │ compute-bound region      │
  │     /.                         │                           │
  │    /                           │                           │
  │   /                                                        │
  │  └──────────────────────────────────────────────────────── │
  │       Arithmetic Intensity (FLOPs / Byte)                  │
  └────────────────────────────────────────────────────────────┘

  Arithmetic Intensity (AI) = FLOPs / Bytes_transferred

  Attainable GFLOPS = min(Peak_GFLOPS, Peak_BW * AI)
     If AI < ridge point → MEMORY-BOUND (optimize memory access)
     If AI > ridge point → COMPUTE-BOUND (optimize ALU usage)

  Ridge point ~ {ridge_point:.1f} FLOP/byte

  Live measurements:
  ──────────────────
  Kernel          AI (FLOP/B)   Time(ms)   GFLOPS   Bottleneck
  ──────────────────────────────────────────────────────────────
  Vector Add      {low_ai:9.3f}   {low_ms:8.3f}   {low_gflops:6.2f}   {"MEMORY-BOUND" if low_ai < ridge_point else "COMPUTE-BOUND"}
  Heavy Compute   {high_ai:9.1f}   {high_ms:8.3f}   {high_gflops:6.2f}   {"MEMORY-BOUND" if high_ai < ridge_point else "COMPUTE-BOUND"}""")

    # Classify common kernels
    print("""
  Common kernel classifications:
  ──────────────────────────────
  Kernel                AI (FLOP/B)   Likely Bottleneck
  ──────────────────────────────────────────────────────
  Vector copy           0.00          MEMORY
  Vector add (SAXPY)    0.125         MEMORY
  Dot product           0.25          MEMORY
  Matrix transpose      0.00          MEMORY
  Histogram             ~0.5          MEMORY
  1D stencil (3-pt)     0.33          MEMORY
  Matrix multiply (NxN) N/12          COMPUTE (for large N)
  Convolution (3x3)     ~1.5          DEPENDS on data size
  Deep NN layer         ~10-100       COMPUTE

  INTERVIEW KEY POINTS:
  ─────────────────────
  • First question: "Is my kernel compute-bound or memory-bound?"
  • Most simple kernels are MEMORY-BOUND on modern GPUs
  • Optimization strategy differs:
    Memory-bound → coalescing, tiling, reduce transfers
    Compute-bound → ILP, vectorize, reduce divergence
  • Matrix multiply is one of the few naturally compute-bound kernels
  • Know the formula: AI = FLOPs / Bytes
""")


# ===================================================================
# 4. DOUBLE BUFFERING — Overlap Transfer with Compute
# ===================================================================
# While GPU processes batch N, CPU prepares batch N+1.
# Uses TWO sets of buffers, alternating each iteration.
#
# Interview Q: "How to maximize throughput in a streaming pipeline?"
# ===================================================================

def demo_double_buffering(ctx, queue, device):
    section("4", "DOUBLE BUFFERING",
            "Overlap data transfer with compute for higher throughput")

    N = 1 << 20  # 1M elements per batch
    NUM_BATCHES = 20
    mf = cl.mem_flags

    KERNEL = """
    __kernel void process(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) {
            float x = in[i];
            for (int k = 0; k < 20; k++)
                x = x * 1.001f + 0.001f;
            out[i] = x;
        }
    }
    """
    prg = cl.Program(ctx, KERNEL).build()

    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  SINGLE BUFFERING (sequential):                              │
  │  [Upload A][Compute A][Download A][Upload B][Compute B]...   │
  │                                                              │
  │  DOUBLE BUFFERING (overlapped):                              │
  │  [Upload A][Compute A ][Download A]                          │
  │            [Upload B  ][Compute B ][Download B]              │
  │                       [Upload C  ][Compute C ][Download C]   │
  │                                                              │
  │  Throughput improvement: up to 2-3x for transfer-heavy work  │
  └──────────────────────────────────────────────────────────────┘
""")

    # --- Single buffering (sequential) ---
    d_in = cl.Buffer(ctx, mf.READ_WRITE, N * 4)
    d_out = cl.Buffer(ctx, mf.READ_WRITE, N * 4)
    out_host = np.empty(N, dtype=np.float32)

    # Use a second queue for async transfers
    queue2 = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    t_single = time.perf_counter()
    for i in range(NUM_BATCHES):
        batch = np.random.randn(N).astype(np.float32)
        cl.enqueue_copy(queue, d_in, batch).wait()
        prg.process(queue, (N,), (256,), d_in, d_out, np.int32(N)).wait()
        cl.enqueue_copy(queue, out_host, d_out).wait()
    single_ms = (time.perf_counter() - t_single) * 1000

    # --- Double buffering (overlapped) ---
    # Two sets of device buffers
    d_in_A = cl.Buffer(ctx, mf.READ_WRITE, N * 4)
    d_out_A = cl.Buffer(ctx, mf.READ_WRITE, N * 4)
    d_in_B = cl.Buffer(ctx, mf.READ_WRITE, N * 4)
    d_out_B = cl.Buffer(ctx, mf.READ_WRITE, N * 4)

    buffers = [(d_in_A, d_out_A), (d_in_B, d_out_B)]
    out_host_A = np.empty(N, dtype=np.float32)
    out_host_B = np.empty(N, dtype=np.float32)

    t_double = time.perf_counter()
    # Prime: upload first batch
    batch0 = np.random.randn(N).astype(np.float32)
    cl.enqueue_copy(queue, d_in_A, batch0)

    for i in range(NUM_BATCHES):
        curr = i % 2
        nxt = (i + 1) % 2
        d_in_curr, d_out_curr = buffers[curr]
        d_in_next, d_out_next = buffers[nxt]

        # Start compute on current batch
        compute_evt = prg.process(queue, (N,), (256,), d_in_curr, d_out_curr,
                                  np.int32(N))

        # While GPU computes, CPU prepares next batch and uploads
        if i < NUM_BATCHES - 1:
            next_batch = np.random.randn(N).astype(np.float32)
            cl.enqueue_copy(queue2, d_in_next, next_batch)

        # Wait for compute to finish, then download result
        compute_evt.wait()
        queue2.finish()
        out_buf = out_host_A if curr == 0 else out_host_B
        cl.enqueue_copy(queue, out_buf, d_out_curr).wait()

    double_ms = (time.perf_counter() - t_double) * 1000

    speedup = single_ms / double_ms if double_ms > 0 else 0

    print(f"  {NUM_BATCHES} batches of {N:,} elements each:")
    print(f"    Single buffering: {single_ms:.1f} ms ({single_ms/NUM_BATCHES:.2f} ms/batch)")
    print(f"    Double buffering: {double_ms:.1f} ms ({double_ms/NUM_BATCHES:.2f} ms/batch)")
    print(f"    Speedup:          {speedup:.2f}x")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • Double buffering = TWO sets of device memory, alternate each frame
  • While GPU processes buffer A, CPU uploads to buffer B
  • Requires TWO command queues (or out-of-order queue)
  • Key insight: overlap is only beneficial when transfer time ~ compute time
  • If compute >> transfer, single buffering is fine (compute dominates)
  • If transfer >> compute, you need to reduce data or batch larger
  • Triple buffering: adds a third buffer for download overlap too
  • Real-world: video processing, streaming inference, sensor pipelines
""")


# ===================================================================
# 5. KERNEL FUSION — Fewer Passes = More Performance
# ===================================================================
# Two separate kernels that read/write global memory
# vs one fused kernel that does both in one pass.
#
# Interview Q: "How do you optimize a sequence of element-wise ops?"
# ===================================================================

def demo_kernel_fusion(ctx, queue, device):
    section("5", "KERNEL FUSION",
            "Combine kernels to reduce memory traffic")

    N = 1 << 22
    mf = cl.mem_flags
    data = np.random.randn(N).astype(np.float32)

    # Separate kernels: normalize then ReLU
    K_SEPARATE = """
    __kernel void normalize(__global const float* in, __global float* out,
                            float mean, float inv_std, int N) {
        int i = get_global_id(0);
        if (i < N) out[i] = (in[i] - mean) * inv_std;
    }
    __kernel void relu(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) out[i] = fmax(0.0f, in[i]);
    }
    """

    # Fused kernel: normalize + ReLU in one pass
    K_FUSED = """
    __kernel void normalize_relu(__global const float* in, __global float* out,
                                 float mean, float inv_std, int N) {
        int i = get_global_id(0);
        if (i < N) out[i] = fmax(0.0f, (in[i] - mean) * inv_std);
    }
    """

    prg_sep = cl.Program(ctx, K_SEPARATE).build()
    prg_fused = cl.Program(ctx, K_FUSED).build()

    mean_val = np.float32(np.mean(data))
    inv_std = np.float32(1.0 / max(np.std(data), 1e-7))

    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_tmp = cl.Buffer(ctx, mf.READ_WRITE, data.nbytes)
    d_out_sep = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)
    d_out_fused = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    # Warmup
    prg_sep.normalize(queue, (N,), (256,), d_in, d_tmp, mean_val, inv_std,
                      np.int32(N)).wait()
    prg_sep.relu(queue, (N,), (256,), d_tmp, d_out_sep, np.int32(N)).wait()
    prg_fused.normalize_relu(queue, (N,), (256,), d_in, d_out_fused, mean_val,
                             inv_std, np.int32(N)).wait()

    # Measure separate
    sep_times = []
    for _ in range(10):
        e1 = prg_sep.normalize(queue, (N,), (256,), d_in, d_tmp, mean_val,
                               inv_std, np.int32(N))
        e2 = prg_sep.relu(queue, (N,), (256,), d_tmp, d_out_sep, np.int32(N))
        e2.wait()
        t = (e1.profile.end - e1.profile.start + e2.profile.end - e2.profile.start) / 1e6
        sep_times.append(t)

    # Measure fused
    fused_times = []
    for _ in range(10):
        e = prg_fused.normalize_relu(queue, (N,), (256,), d_in, d_out_fused,
                                     mean_val, inv_std, np.int32(N))
        e.wait()
        fused_times.append((e.profile.end - e.profile.start) / 1e6)

    sep_ms = np.mean(sep_times)
    fused_ms = np.mean(fused_times)

    # Memory traffic analysis
    sep_bytes = N * 4 * 4  # read in, write tmp, read tmp, write out
    fused_bytes = N * 4 * 2  # read in, write out
    sep_bw = sep_bytes / (sep_ms / 1000) / 1e9
    fused_bw = fused_bytes / (fused_ms / 1000) / 1e9

    print(f"\n  Normalize + ReLU on {N:,} elements:\n")
    print(f"  {'Approach':>18s} {'Time(ms)':>10s} {'Mem Traffic':>12s} {'BW(GB/s)':>10s}")
    print("  " + "-" * 55)
    print(f"  {'Separate (2 pass)':>18s} {sep_ms:>10.3f} {sep_bytes/1e6:>10.1f} MB {sep_bw:>10.1f}")
    print(f"  {'Fused (1 pass)':>18s} {fused_ms:>10.3f} {fused_bytes/1e6:>10.1f} MB {fused_bw:>10.1f}")
    print(f"  {'Speedup':>18s} {sep_ms/fused_ms:>10.2f}x {sep_bytes/fused_bytes:>10.1f}x")

    # Verify correctness
    out_sep = np.empty(N, dtype=np.float32)
    out_fused = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(queue, out_sep, d_out_sep).wait()
    cl.enqueue_copy(queue, out_fused, d_out_fused).wait()
    match = np.allclose(out_sep, out_fused, atol=1e-6)

    print(f"\n  Results match: {match}")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • Fusion eliminates intermediate global memory read/write
  • For memory-bound kernels, fusion can give 2x+ speedup
  • ONNX Runtime does this automatically (graph optimization)
  • Manual fusion is needed for custom OpenCL kernels
  • Trade-off: fused kernels are more complex, harder to debug
  • Rule: if two kernels are both memory-bound and element-wise, FUSE them
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 6: PERFORMANCE THEORY & SCALING LAWS")
    print("  The theory interviewers ALWAYS ask about")
    print("=" * 70)

    ctx, queue, device = get_gpu_context()
    print(f"  GPU: {device.name} ({device.max_compute_units} CUs)")

    demo_amdahls_law(ctx, queue, device)
    demo_gustafsons_law(ctx, queue, device)
    demo_roofline(ctx, queue, device)
    demo_double_buffering(ctx, queue, device)
    demo_kernel_fusion(ctx, queue, device)

    print("\n" + "=" * 70)
    print("  SESSION 6 COMPLETE — PERFORMANCE THEORY SUMMARY")
    print("=" * 70)
    print("""
  Concepts learned:
  ─────────────────
  1. Amdahl's Law:    Speedup = 1/(S + (1-S)/P) — serial fraction limits you
  2. Gustafson's Law: Scaled_Speedup = P - S*(P-1) — scale the problem
  3. Roofline Model:  AI = FLOPs/Bytes → compute vs memory bound
  4. Double Buffering: Overlap transfer + compute with two buffer sets
  5. Kernel Fusion:   Fewer memory passes = less bandwidth pressure

  When asked "how would you optimize this?":
  ──────────────────────────────────────────
  Step 1: Is it compute-bound or memory-bound? (Roofline)
  Step 2: What's the serial fraction? (Amdahl)
  Step 3: Can you scale the problem? (Gustafson)
  Step 4: Can you overlap transfer and compute? (Double buffering)
  Step 5: Can you fuse kernels? (Kernel fusion)

  Next: Session 7 — Advanced algorithms (Blelloch scan, bitonic sort, ...)
""")


if __name__ == "__main__":
    main()
