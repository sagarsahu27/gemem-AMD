"""
Session 0: Core Concepts — How Hardware Shapes Your Code
========================================================
Before writing GPU/NPU kernels, understand the building blocks.
Each concept directly impacts how you configure and tune your code.

This session is REFERENCE material — run it to see live measurements
on YOUR hardware, then revisit as you work through Sessions 1-5.

Sections:
  0. Execution Model     — How work is launched and scheduled
  1. Memory Hierarchy    — Where data lives and how fast it moves
  2. Work-Group Config   — Choosing sizes and why it matters
  3. Wavefronts & SIMD   — AMD's RDNA execution unit
  4. Occupancy           — Keeping the GPU busy
  5. Data Transfer Cost  — Host ↔ Device movement
  6. ONNX RT Providers   — Execution provider chain and session config
  7. Environment Config  — Firmware, cache, env vars that control behavior

Hardware: AMD Radeon 860M (RDNA 3.5) + XDNA 2 NPU
Run: conda run -n ryzen-ai-1.7.1 --no-capture-output python tutorials/session0_concepts.py
"""

import numpy as np
import pyopencl as cl
import onnxruntime as ort
import os
import time
import sys


# ===================================================================
# HELPERS
# ===================================================================

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
    print(f"  CONCEPT {num}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70)


# ===================================================================
# CONCEPT 0: EXECUTION MODEL — How Work is Launched
# ===================================================================
# Your code runs in two places:
#   HOST  = CPU (Python, C++) — controls everything
#   DEVICE = GPU or NPU — does the heavy lifting
#
# The host ENQUEUES work onto the device and (optionally) waits.
# ===================================================================

def concept_execution_model(ctx, queue, device):
    section(0, "EXECUTION MODEL",
            "Host (CPU) enqueues work → Device (GPU/NPU) executes")

    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │                    EXECUTION FLOW                           │
  │                                                             │
  │  HOST (CPU / Python)              DEVICE (GPU / NPU)       │
  │  ───────────────────              ──────────────────        │
  │                                                             │
  │  1. Allocate buffers ──────────→  Device memory reserved    │
  │  2. Write data       ──────────→  Host→Device transfer      │
  │  3. Enqueue kernel   ──────────→  Kernel executes           │
  │     (returns immediately)         (async on device)         │
  │  4. Wait / read back ←──────────  Device→Host transfer      │
  │                                                             │
  │  KEY: Steps 1-3 are ASYNCHRONOUS.                           │
  │       The CPU can do other work while GPU computes.         │
  │       Only step 4 (read back) forces synchronization.       │
  └─────────────────────────────────────────────────────────────┘

  Impact on your code:
  ─────────────────────
  • cl.enqueue_nd_range_kernel() returns IMMEDIATELY
    → The kernel is running on the GPU while Python continues
  • evt.wait() or cl.enqueue_copy(...).wait() BLOCKS until done
  • For pipelines: overlap CPU work with GPU execution
  • For latency: minimize synchronization points
""")

    # Live demo: async vs sync timing
    N = 1 << 20
    data = np.random.randn(N).astype(np.float32)
    mf = cl.mem_flags

    KERNEL = """
    __kernel void square(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) out[i] = in[i] * in[i];
    }
    """
    prg = cl.Program(ctx, KERNEL).build()
    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    kernel = cl.Kernel(prg, "square")
    kernel.set_args(d_in, d_out, np.int32(N))

    # Measure: enqueue (non-blocking) vs wait (blocking)
    t0 = time.perf_counter()
    evt = cl.enqueue_nd_range_kernel(queue, kernel, (N,), None)
    t_enqueue = time.perf_counter() - t0

    t0 = time.perf_counter()
    evt.wait()
    t_wait = time.perf_counter() - t0

    gpu_us = (evt.profile.end - evt.profile.start) / 1000

    print(f"  Live measurement ({N:,} elements):")
    print(f"    Enqueue call:     {t_enqueue*1e6:8.1f} us  (returns immediately)")
    print(f"    evt.wait() call:  {t_wait*1e6:8.1f} us  (blocks until GPU done)")
    print(f"    Actual GPU time:  {gpu_us:8.1f} us  (from hardware profiling)")
    print(f"    CPU was FREE for: {gpu_us - t_enqueue*1e6:.0f} us while GPU worked\n")


# ===================================================================
# CONCEPT 1: MEMORY HIERARCHY — Where Data Lives
# ===================================================================
# The #1 performance factor. Not compute — MEMORY ACCESS.
#
# Fastest → Slowest:
#   Private registers > Local (LDS) > Global (VRAM) > Host (DDR)
#
# Your code configuration: choose WHERE to put data.
# ===================================================================

def concept_memory_hierarchy(ctx, queue, device):
    section(1, "MEMORY HIERARCHY",
            "Performance = memory access patterns, not just compute")

    lds_kb = device.local_mem_size // 1024
    global_mb = device.global_mem_size // (1024 * 1024)
    max_alloc_mb = device.max_mem_alloc_size // (1024 * 1024)

    print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  MEMORY TYPE     │ SCOPE          │ SPEED   │ YOUR DEVICE    │
  │──────────────────│────────────────│─────────│────────────────│
  │  Private (regs)  │ 1 work-item    │ ~0.5 ns │ per-thread     │
  │  __local (LDS)   │ 1 work-group   │ ~5 ns   │ {lds_kb:4d} KB total   │
  │  __global (VRAM) │ all work-items │ ~100 ns │ {global_mb:4d} MB total  │
  │  Host (DDR)      │ CPU only       │ ~200 ns │ system RAM     │
  └──────────────────────────────────────────────────────────────┘

  Max single allocation: {max_alloc_mb} MB

  In your code, you DECLARE where data lives:
  ─────────────────────────────────────────────
  __global float* data     →  Device VRAM (large, slow)
  __local  float  tile[64] →  Work-group shared LDS (small, fast)
  float temp               →  Private register (tiny, fastest)

  Impact on code configuration:
  ─────────────────────────────
  • Tiling patterns use __local to cache __global data
  • Reduce patterns accumulate in __local, write 1 result to __global
  • Stencil patterns load neighborhood into __local (halo cells)
  • cl.LocalMemory(size) allocates LDS at kernel launch time
""")

    # Measure: global-only vs local-tiled access
    N = 1 << 20
    data = np.random.randn(N).astype(np.float32)
    mf = cl.mem_flags
    d_data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, 4)

    # Global-only reduce
    K_GLOBAL = """
    __kernel void reduce_global(__global const float* data,
                                __global float* result, int N) {
        // BAD: reads from __global memory log(N) times per element
        float sum = 0.0f;
        for (int i = get_global_id(0); i < N; i += get_global_size(0))
            sum += data[i];
        // Just write to avoid optimizing away
        if (get_global_id(0) == 0) *result = sum;
    }
    """
    # Local-tiled reduce
    K_LOCAL = """
    __kernel void reduce_local(__global const float* data,
                               __local float* scratch,
                               __global float* result, int N) {
        int lid = get_local_id(0);
        int gs = get_local_size(0);
        float acc = 0.0f;
        for (int i = get_global_id(0); i < N; i += get_global_size(0))
            acc += data[i];
        scratch[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = gs/2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) *result = scratch[0];
    }
    """
    prg_g = cl.Program(ctx, K_GLOBAL).build()
    prg_l = cl.Program(ctx, K_LOCAL).build()

    LS = 256
    GS = LS * 4  # 4 work-groups

    # Warmup + measure global
    prg_g.reduce_global(queue, (GS,), (LS,), d_data, d_out, np.int32(N)).wait()
    evt_g = prg_g.reduce_global(queue, (GS,), (LS,), d_data, d_out, np.int32(N))
    evt_g.wait()
    t_global = (evt_g.profile.end - evt_g.profile.start) / 1e6

    # Warmup + measure local
    prg_l.reduce_local(queue, (GS,), (LS,), d_data, cl.LocalMemory(LS * 4),
                       d_out, np.int32(N)).wait()
    evt_l = prg_l.reduce_local(queue, (GS,), (LS,), d_data, cl.LocalMemory(LS * 4),
                               d_out, np.int32(N))
    evt_l.wait()
    t_local = (evt_l.profile.end - evt_l.profile.start) / 1e6

    print(f"  Live measurement (reduce {N:,} floats):")
    print(f"    __global only:  {t_global:.3f} ms")
    print(f"    __local tiled:  {t_local:.3f} ms")
    if t_local < t_global:
        print(f"    Speedup:        {t_global/t_local:.1f}x from using local memory\n")
    else:
        print(f"    Note: small kernel — overhead dominates at this scale\n")


# ===================================================================
# CONCEPT 2: WORK-GROUP CONFIGURATION
# ===================================================================
# When you launch a kernel, you choose:
#   Global size: total number of work-items
#   Local size:  work-items per work-group (= threads per block)
#
# This is the MOST IMPORTANT configuration decision.
# ===================================================================

def concept_workgroup_config(ctx, queue, device):
    section(2, "WORK-GROUP CONFIGURATION",
            "global_size and local_size control parallelism")

    max_wg = device.max_work_group_size
    max_dims = device.max_work_item_sizes
    cus = device.max_compute_units

    print(f"""
  Your GPU limits:
  ────────────────
  Max work-group size:  {max_wg}
  Max dimensions:       {max_dims[0]} x {max_dims[1]} x {max_dims[2]}
  Compute Units (CUs):  {cus}

  How it maps to your kernel launch:
  ──────────────────────────────────
  cl.enqueue_nd_range_kernel(queue, kernel,
      global_size=(1024,),    # Total work-items to launch
      local_size=(256,))      # Work-items per work-group

  This creates: 1024 / 256 = 4 work-groups, each with 256 items.

  ┌──────────────────────────────────────────────────────────┐
  │  global_size = (1024,)                                   │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
  │  │  WG 0    │ │  WG 1    │ │  WG 2    │ │  WG 3    │    │
  │  │ 256 items│ │ 256 items│ │ 256 items│ │ 256 items│    │
  │  │ share LDS│ │ share LDS│ │ share LDS│ │ share LDS│    │
  │  │ barrier()│ │ barrier()│ │ barrier()│ │ barrier()│    │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
  │  Each WG runs on ONE Compute Unit.                       │
  │  WGs do NOT synchronize with each other.                 │
  └──────────────────────────────────────────────────────────┘

  Rules of thumb for local_size:
  ──────────────────────────────
  • Must be ≤ {max_wg} (your GPU's max)
  • Should be a multiple of wavefront size (32 or 64 on AMD)
  • Common choices: 64, 128, 256
  • Larger → more shared memory per group, more barrier overhead
  • Smaller → less LDS usage, more groups can run concurrently
  • None → OpenCL runtime picks (often suboptimal!)
""")

    # Benchmark different work-group sizes
    N = 1 << 22  # 4M elements
    data = np.random.randn(N).astype(np.float32)
    mf = cl.mem_flags
    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    KERNEL = """
    __kernel void saxpy(__global const float* X, __global float* Y, int N) {
        int i = get_global_id(0);
        if (i < N) Y[i] = 2.0f * X[i] + 1.0f;
    }
    """
    prg = cl.Program(ctx, KERNEL).build()

    print(f"  Benchmarking local_size on saxpy ({N:,} elements):")
    print(f"  {'local_size':>12s} {'WGs':>6s} {'Kernel (ms)':>12s} {'GB/s':>8s}")
    print("  " + "-" * 44)

    for ls in [32, 64, 128, 256]:
        gs = ((N + ls - 1) // ls) * ls
        num_wgs = gs // ls

        # Warmup
        prg.saxpy(queue, (gs,), (ls,), d_in, d_out, np.int32(N)).wait()

        # Timed (average 5 runs)
        times = []
        for _ in range(5):
            evt = prg.saxpy(queue, (gs,), (ls,), d_in, d_out, np.int32(N))
            evt.wait()
            times.append((evt.profile.end - evt.profile.start) / 1e6)
        avg_ms = np.mean(times)
        gbps = (N * 4 * 2) / (avg_ms / 1000) / 1e9  # read + write

        print(f"  {ls:>12d} {num_wgs:>6,d} {avg_ms:>12.3f} {gbps:>8.1f}")

    print()


# ===================================================================
# CONCEPT 3: WAVEFRONTS & SIMD — AMD's Execution Unit
# ===================================================================
# AMD GPUs execute work-items in groups of 32 or 64 called WAVEFRONTS.
# (NVIDIA calls them "warps" — same concept, size = 32.)
#
# All items in a wavefront execute THE SAME instruction simultaneously.
# If they diverge (if/else), both paths execute sequentially → slowdown.
# ===================================================================

def concept_wavefronts(ctx, queue, device):
    section(3, "WAVEFRONTS & SIMD",
            "AMD executes in lock-step groups of 32/64 work-items")

    # Detect wavefront size
    try:
        wavefront = device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)
    except Exception:
        wavefront = 64  # Default for AMD

    print(f"""
  Your GPU wavefront size: {wavefront}

  What this means:
  ────────────────
  The GPU groups {wavefront} work-items into a WAVEFRONT.
  All {wavefront} items execute the SAME instruction at the SAME time.
  This is SIMD: Single Instruction, Multiple Data.

  ┌─────────────────────────────────────────────────────┐
  │  Wavefront (size={wavefront})                               │
  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬──────┐  │
  │  │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │ ... │  │
  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴──────┘  │
  │  ALL execute: Y[i] = A * X[i] + B  simultaneously   │
  └─────────────────────────────────────────────────────┘

  DIVERGENCE PROBLEM:
  ───────────────────
  if (get_global_id(0) % 2 == 0)
      doA();     ← Even threads execute, odd threads IDLE
  else
      doB();     ← Odd threads execute, even threads IDLE

  Both branches execute SEQUENTIALLY → 50% efficiency!

  Impact on your code:
  ────────────────────
  • Minimize divergent if/else inside kernels
  • Ensure all items in a wavefront take the same path
  • Boundary checks (if i < N) only affect the last wavefront → OK
  • Work-group size should be multiple of {wavefront}
""")

    # Demonstrate divergence cost
    N = 1 << 22
    data = np.random.randn(N).astype(np.float32)
    mf = cl.mem_flags
    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    K_UNIFORM = """
    __kernel void uniform(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) {
            // ALL work-items take the same path (uniform)
            out[i] = in[i] * 2.0f + 1.0f;
        }
    }
    """

    K_DIVERGENT = """
    __kernel void divergent(__global const float* in, __global float* out, int N) {
        int i = get_global_id(0);
        if (i < N) {
            // ALTERNATING work-items take different paths (divergent)
            if (i % 2 == 0)
                out[i] = in[i] * 2.0f + 1.0f;
            else
                out[i] = in[i] * 0.5f - 1.0f;
        }
    }
    """

    prg_u = cl.Program(ctx, K_UNIFORM).build()
    prg_d = cl.Program(ctx, K_DIVERGENT).build()
    LS = 256

    # Warmup
    prg_u.uniform(queue, (N,), (LS,), d_in, d_out, np.int32(N)).wait()
    prg_d.divergent(queue, (N,), (LS,), d_in, d_out, np.int32(N)).wait()

    times_u, times_d = [], []
    for _ in range(10):
        e = prg_u.uniform(queue, (N,), (LS,), d_in, d_out, np.int32(N))
        e.wait()
        times_u.append((e.profile.end - e.profile.start) / 1e6)
        e = prg_d.divergent(queue, (N,), (LS,), d_in, d_out, np.int32(N))
        e.wait()
        times_d.append((e.profile.end - e.profile.start) / 1e6)

    avg_u = np.mean(times_u)
    avg_d = np.mean(times_d)
    print(f"  Live measurement ({N:,} elements, local_size={LS}):")
    print(f"    Uniform kernel:   {avg_u:.3f} ms  (all same path)")
    print(f"    Divergent kernel: {avg_d:.3f} ms  (alternating if/else)")
    print(f"    Divergence cost:  {avg_d/avg_u:.2f}x\n")


# ===================================================================
# CONCEPT 4: OCCUPANCY — Keeping the GPU Busy
# ===================================================================
# Occupancy = active wavefronts / max possible wavefronts per CU.
# Higher occupancy → GPU can hide memory latency by switching waves.
#
# What limits occupancy: register usage, LDS usage, work-group size.
# ===================================================================

def concept_occupancy(ctx, queue, device):
    section(4, "OCCUPANCY",
            "Active wavefronts / max wavefronts — hiding memory latency")

    cus = device.max_compute_units
    max_wg = device.max_work_group_size
    lds_total = device.local_mem_size

    print(f"""
  Your GPU: {cus} Compute Units, {lds_total//1024} KB LDS per CU

  ┌──────────────────────────────────────────────────────────┐
  │  OCCUPANCY: How well you're utilizing the GPU             │
  │                                                           │
  │  Each CU can run MULTIPLE wavefronts simultaneously.     │
  │  When one wavefront stalls on memory, another executes.  │
  │                                                           │
  │  HIGH occupancy → good latency hiding → better perf      │
  │  LOW occupancy  → GPU idles during memory stalls         │
  └──────────────────────────────────────────────────────────┘

  What LIMITS occupancy:
  ──────────────────────

  1. REGISTERS per work-item
     • More registers per item → fewer concurrent wavefronts
     • Complex kernels with many local variables → low occupancy
     → Fix: reduce variable count, break into simpler kernels

  2. LOCAL MEMORY (LDS) per work-group
     • More LDS → fewer work-groups fit on a CU
     • Large tiles, big local arrays → low occupancy
     → Fix: smaller tiles, or process in multiple passes

  3. WORK-GROUP SIZE
     • Too small → not enough wavefronts per group
     • Too large → fewer groups fit on each CU
     → Fix: benchmark different sizes (Concept 2)

  Trade-off:
  ──────────
  • Tiled MatMul uses 16×16 = 256 threads, 16×16×4 = 1KB LDS per tile
    → Moderate occupancy, but each access is FAST (LDS vs VRAM)
  • Sometimes LOWER occupancy with MORE LDS is faster!
  • The only way to know: MEASURE (see Concept 2 benchmark)

  Impact on your code:
  ────────────────────
  • cl.LocalMemory(N) → you control LDS allocation per work-group
  • More temp variables → compiler uses more registers
  • Keep kernels focused: one kernel, one task
  • Profile with rocprof (Linux) or Radeon GPU Profiler
""")

    # Show LDS usage impact
    N = 1 << 22
    data = np.random.randn(N).astype(np.float32)
    mf = cl.mem_flags
    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_out = cl.Buffer(ctx, mf.WRITE_ONLY, 256 * 4)

    K_SMALL_LDS = """
    __kernel void reduce_small(__global const float* data,
                               __local float* s, __global float* out, int N) {
        int lid = get_local_id(0);
        float acc = 0;
        for (int i = get_global_id(0); i < N; i += get_global_size(0))
            acc += data[i];
        s[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int st = get_local_size(0)/2; st > 0; st >>= 1) {
            if (lid < st) s[lid] += s[lid + st];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) out[get_group_id(0)] = s[0];
    }
    """
    prg = cl.Program(ctx, K_SMALL_LDS).build()

    print(f"  LDS usage vs kernel time ({N:,} elements, 256 work-groups):")
    print(f"  {'WG Size':>10s} {'LDS/WG':>10s} {'Kernel(ms)':>12s}")
    print("  " + "-" * 36)

    for ls in [64, 128, 256]:
        gs = ls * 256
        lds_bytes = ls * 4  # float per work-item
        prg.reduce_small(queue, (gs,), (ls,), d_in,
                         cl.LocalMemory(lds_bytes), d_out, np.int32(N)).wait()
        times = []
        for _ in range(5):
            e = prg.reduce_small(queue, (gs,), (ls,), d_in,
                                 cl.LocalMemory(lds_bytes), d_out, np.int32(N))
            e.wait()
            times.append((e.profile.end - e.profile.start) / 1e6)
        print(f"  {ls:>10d} {lds_bytes:>8d} B {np.mean(times):>12.3f}")

    print()


# ===================================================================
# CONCEPT 5: DATA TRANSFER COST — Host ↔ Device
# ===================================================================
# Moving data between CPU and GPU is EXPENSIVE.
# If your compute is small, the transfer dominates.
# This is why small arrays are faster on CPU.
# ===================================================================

def concept_data_transfer(ctx, queue, device):
    section(5, "DATA TRANSFER COST",
            "Host ↔ Device copies often dominate total time")

    mf = cl.mem_flags

    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  Transfer Timeline:                                          │
  │                                                              │
  │  CPU RAM ──[ upload ]──→ GPU VRAM ──[ compute ]──→ result    │
  │                                    ──[ download ]──→ CPU RAM │
  │                                                              │
  │  For iGPU (like yours): CPU and GPU share system DDR,        │
  │  so "transfer" is cheaper than discrete GPU (no PCIe).       │
  │  But the overhead of buffer creation/mapping still exists.   │
  └──────────────────────────────────────────────────────────────┘
""")

    sizes = [1024, 16384, 262144, 1048576, 4194304, 16777216]

    print(f"  {'Size':>12s} {'Upload(ms)':>12s} {'Download(ms)':>12s} {'Total(ms)':>12s} {'BW(GB/s)':>10s}")
    print("  " + "-" * 62)

    for N in sizes:
        data = np.random.randn(N).astype(np.float32)
        nbytes = N * 4
        out = np.empty(N, dtype=np.float32)

        # Measure upload (host → device)
        times_up = []
        for _ in range(5):
            t0 = time.perf_counter()
            buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
            queue.finish()
            times_up.append(time.perf_counter() - t0)

        # Measure download (device → host)
        buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)
        times_down = []
        for _ in range(5):
            t0 = time.perf_counter()
            cl.enqueue_copy(queue, out, buf).wait()
            times_down.append(time.perf_counter() - t0)

        up_ms = np.mean(times_up) * 1000
        down_ms = np.mean(times_down) * 1000
        total_ms = up_ms + down_ms
        bw = (nbytes * 2) / (total_ms / 1000) / 1e9  # round-trip bandwidth

        size_str = f"{nbytes:,} B" if nbytes < 1e6 else f"{nbytes/1e6:.1f} MB"
        print(f"  {size_str:>12s} {up_ms:>12.3f} {down_ms:>12.3f} {total_ms:>12.3f} {bw:>10.1f}")

    print("""
  Impact on your code:
  ────────────────────
  • For small data: transfer overhead > compute savings → use CPU
  • For large data: compute savings >> transfer → use GPU
  • Reuse device buffers — don't re-upload unchanged data
  • Batch multiple operations between one upload and one download
  • iGPU (shared memory) has lower transfer cost than discrete GPU
""")


# ===================================================================
# CONCEPT 6: ONNX RUNTIME EXECUTION PROVIDERS
# ===================================================================
# ONNX Runtime doesn't run "on GPU" — it routes ops to PROVIDERS.
# The provider chain determines which device runs each graph node.
# ===================================================================

def concept_execution_providers():
    section(6, "ONNX RT EXECUTION PROVIDERS",
            "Provider chain controls where each op runs")

    available = ort.get_available_providers()

    print(f"""
  Available Execution Providers on your system:
  ──────────────────────────────────────────────""")
    for i, ep in enumerate(available):
        device_map = {
            "VitisAIExecutionProvider": "NPU (XDNA 2 AIE tiles)",
            "DmlExecutionProvider": "GPU (DirectX ML on Radeon 860M)",
            "CPUExecutionProvider": "CPU (Zen 5 cores)",
        }
        desc = device_map.get(ep, "Unknown device")
        print(f"    {i+1}. {ep:40s} → {desc}")

    print(f"""
  How the provider chain works:
  ─────────────────────────────
  sess = ort.InferenceSession("model.onnx", providers=[
      "VitisAIExecutionProvider",   # Priority 1: try NPU first
      "DmlExecutionProvider",       # Priority 2: fallback to GPU
      "CPUExecutionProvider",       # Priority 3: always available
  ])

  For EACH node in the ONNX graph:
    1. Ask VitisAI EP: "Can you run this MatMul?" → Yes? → NPU
    2. If not → Ask DML EP: "Can you run this op?" → Yes? → GPU
    3. If not → CPU EP runs it (always says yes)

  ┌──────────────────────────────────────────────────────────┐
  │  ONNX Graph:  Input → Conv → Relu → MatMul → Softmax   │
  │                         │       │       │        │       │
  │  VitisAI EP:           YES     YES     YES      NO      │
  │  DML EP:               YES     YES     YES     YES      │
  │  CPU EP:               YES     YES     YES     YES      │
  │                                                          │
  │  With VitisAI first:  [NPU, NPU, NPU, CPU fallback]    │
  │  With DML first:      [GPU, GPU, GPU, GPU]              │
  └──────────────────────────────────────────────────────────┘

  Key session options that affect behavior:
  ─────────────────────────────────────────""")

    so = ort.SessionOptions()
    print(f"    graph_optimization_level: {so.graph_optimization_level}")
    print(f"    inter_op_num_threads:     {so.inter_op_num_threads} (0=auto)")
    print(f"    intra_op_num_threads:     {so.intra_op_num_threads} (0=auto)")
    print(f"    enable_profiling:         {so.enable_profiling}")

    print("""
  Impact on your code:
  ────────────────────
  • Provider ORDER matters — first capable provider wins
  • VitisAI EP: best for quantized models, batch=1, power-efficient
  • DML EP: best for large batch, FP16/FP32, GPU acceleration
  • CPU EP: best for dynamic shapes, unsupported ops, debugging
  • Session creation is EXPENSIVE (compilation) — reuse sessions
  • VitisAI EP caches compiled models in $TEMP/vaip/.cache/
""")


# ===================================================================
# CONCEPT 7: ENVIRONMENT CONFIGURATION
# ===================================================================
# Several env vars and paths control GPU/NPU behavior.
# Wrong config = silent fallback to CPU or cryptic errors.
# ===================================================================

def concept_environment_config():
    section(7, "ENVIRONMENT CONFIGURATION",
            "Env vars, firmware, and paths that control device behavior")

    firmware = os.environ.get("XLNX_VART_FIRMWARE", "<NOT SET>")
    vaip_cache = os.path.join(os.environ.get("TEMP", ""), "vaip", ".cache")
    cache_exists = os.path.isdir(vaip_cache)

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  ENVIRONMENT VARIABLE           │  PURPOSE                  │
  │─────────────────────────────────│───────────────────────────│
  │  XLNX_VART_FIRMWARE             │  Path to NPU firmware     │
  │  TEMP / vaip/.cache             │  VitisAI compiled model   │
  │  PYTHONIOENCODING               │  Unicode output (utf-8)   │
  │  NUM_OF_DPU_RUNNERS             │  NPU parallel runners     │
  │  XLNX_ENABLE_CACHE              │  Compilation caching      │
  └─────────────────────────────────────────────────────────────┘

  Your current configuration:
  ───────────────────────────
  XLNX_VART_FIRMWARE:   {firmware}
  VitisAI cache:        {vaip_cache}
  Cache exists:         {cache_exists}
  Python encoding:      {sys.stdout.encoding}""")

    # Check firmware
    firmware_ok = os.path.exists(firmware) if firmware != "<NOT SET>" else False
    xclbin_path = r"C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
    xclbin_exists = os.path.exists(xclbin_path)

    print(f"  Firmware file exists: {firmware_ok}")
    print(f"  Expected xclbin:     {xclbin_exists} ({xclbin_path})")

    print(f"""
  Setting up correctly in your code:
  ───────────────────────────────────
  # Python — set at start of script
  import os
  if not os.environ.get("XLNX_VART_FIRMWARE"):
      os.environ["XLNX_VART_FIRMWARE"] = (
          r"C:\\Windows\\System32\\AMD\\"
          r"AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
      )

  # PowerShell — set before running
  $env:XLNX_VART_FIRMWARE = "C:\\Windows\\System32\\AMD\\AMD_AIE2P_..."

  # Conda activation (permanent for env):
  conda env config vars set XLNX_VART_FIRMWARE="C:\\Windows\\..."

  Impact on your code:
  ────────────────────
  • Missing XLNX_VART_FIRMWARE → VitisAI EP silently fails to load
    → falls back to CPU, no error message!
  • First VitisAI run compiles model → SLOW (30-60s)
    Subsequent runs use cache → FAST (< 1s)
  • Clear cache (delete vaip/.cache) if model changes
  • PYTHONIOENCODING=utf-8 avoids Unicode errors in conda output

  Conda environment: ryzen-ai-1.7.1
  ──────────────────────────────────
  This env was created by the Ryzen AI SDK 1.7.1 installer.
  It contains all 21 SDK wheels pre-installed:
    onnxruntime-vitisai, voe, olive-ai, onnx, torch, ...
  ALWAYS use this env for NPU work:
    conda run -n ryzen-ai-1.7.1 --no-capture-output python <script>
""")

    # Show cache size
    if cache_exists:
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(vaip_cache):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
                file_count += 1
        print(f"  VitisAI cache stats: {file_count} files, {total_size/1e6:.1f} MB\n")
    else:
        print(f"  VitisAI cache: not yet created (runs on first NPU inference)\n")


# ===================================================================
# SUMMARY: CONFIGURATION CHECKLIST
# ===================================================================

def print_checklist():
    print("\n" + "=" * 70)
    print("  CONFIGURATION CHECKLIST — Before Writing Any Kernel")
    print("=" * 70)
    print("""
  GPU (OpenCL) Kernel Checklist:
  ══════════════════════════════
  □ Choose local_size (multiple of wavefront size: 32/64)
  □ Choose global_size (covers all elements, divisible by local_size)
  □ Decide memory types: __global for I/O, __local for shared tiles
  □ Add barrier(CLK_LOCAL_MEM_FENCE) after every __local write
  □ Handle boundary: if (gid < N) for last work-group
  □ Minimize divergence: avoid per-item if/else inside wavefronts
  □ Coalesce memory: adjacent work-items → adjacent addresses
  □ Reuse buffers: don't re-create cl.Buffer() in loops

  NPU (ONNX Runtime) Session Checklist:
  ══════════════════════════════════════
  □ Set XLNX_VART_FIRMWARE env var before creating session
  □ Set model ir_version = 8 (VitisAI supports ≤ 11)
  □ Use numpy < 2.0 (onnxruntime-vitisai compiled against 1.x)
  □ Order providers: VitisAI → DML → CPU (priority order)
  □ Create session ONCE, reuse for all inferences (compilation is slow)
  □ First run triggers compilation (30-60s) — cached after that
  □ Check active providers: sess.get_providers()

  Performance Tuning Order:
  ═════════════════════════
  1. Pick the right DEVICE (CPU vs GPU vs NPU — Concept 5 crossover)
  2. Fix MEMORY ACCESS patterns (coalescing, tiling — Concept 1)
  3. Tune WORK-GROUP SIZE (benchmark 64/128/256 — Concept 2)
  4. Reduce DIVERGENCE (uniform control flow — Concept 3)
  5. Optimize OCCUPANCY (balance LDS vs parallelism — Concept 4)
  6. Minimize TRANSFERS (batch work, reuse buffers — Concept 5)
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 0: CORE CONCEPTS")
    print("  How Hardware Components Shape Your Code Configuration")
    print("=" * 70)

    # Auto-set firmware
    if not os.environ.get("XLNX_VART_FIRMWARE"):
        xclbin = r"C:\Windows\System32\AMD\AMD_AIE2P_Nx4_Overlay_3.5.0.0-2353_ipu_2.xclbin"
        if os.path.exists(xclbin):
            os.environ["XLNX_VART_FIRMWARE"] = xclbin

    ctx, queue, device = get_gpu_context()
    print(f"  GPU: {device.name}")
    print(f"  CUs: {device.max_compute_units}")
    print(f"  Max WG: {device.max_work_group_size}")
    print(f"  LDS: {device.local_mem_size // 1024} KB")
    print(f"  VRAM: {device.global_mem_size // (1024*1024)} MB")
    print(f"  ONNX RT: {', '.join(ort.get_available_providers())}")

    concept_execution_model(ctx, queue, device)
    concept_memory_hierarchy(ctx, queue, device)
    concept_workgroup_config(ctx, queue, device)
    concept_wavefronts(ctx, queue, device)
    concept_occupancy(ctx, queue, device)
    concept_data_transfer(ctx, queue, device)
    concept_execution_providers()
    concept_environment_config()
    print_checklist()

    print("=" * 70)
    print("  SESSION 0 COMPLETE")
    print("  Now proceed to Session 1 with confidence!")
    print("=" * 70)


if __name__ == "__main__":
    main()
