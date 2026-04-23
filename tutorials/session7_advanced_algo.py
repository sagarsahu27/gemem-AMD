"""
Session 7: Advanced Algorithms & Concurrency Patterns
=====================================================
Algorithms and patterns that were identified as gaps.

Topics:
  1. Blelloch (Work-Efficient) Scan — O(N) work, exclusive prefix sum
  2. Multi-Block Scan — scan millions of elements across work-groups
  3. Bitonic Sort — data-oblivious parallel sorting network
  4. Producer-Consumer — bounded buffer with synchronization
  5. Concurrency Hazards — deadlock, false sharing, CAS

Hardware: AMD Radeon 860M (RDNA 3.5) + Zen 5 CPU
Run: conda run -n ryzen-ai-1.7.1 --no-capture-output python tutorials/session7_advanced_algo.py
"""

import numpy as np
import pyopencl as cl
import time
import threading
import queue as stdlib_queue


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
# 1. BLELLOCH (WORK-EFFICIENT) SCAN
# ===================================================================
# Hillis-Steele: O(N log N) work, O(log N) steps — FAST but wasteful
# Blelloch:      O(N) work,      O(log N) steps — work-EFFICIENT
#
# Two phases:
#   Up-sweep (reduce): build partial sums in a tree
#   Down-sweep: distribute prefix sums back down
#
# Produces EXCLUSIVE scan (output[0] = 0)
# ===================================================================

BLELLOCH_KERNEL = """
// Blelloch exclusive scan — work-efficient O(N) algorithm
// Phase 1: Up-sweep (reduce)
__kernel void upsweep(__global int* data, int stride, int N) {
    int i = get_global_id(0);
    int idx = (i + 1) * stride * 2 - 1;
    if (idx < N) {
        data[idx] += data[idx - stride];
    }
}

// Phase 2: Down-sweep
__kernel void downsweep(__global int* data, int stride, int N) {
    int i = get_global_id(0);
    int idx = (i + 1) * stride * 2 - 1;
    if (idx < N) {
        int temp = data[idx - stride];
        data[idx - stride] = data[idx];
        data[idx] += temp;
    }
}

// Set last element to 0 (identity for exclusive scan)
__kernel void set_zero(__global int* data, int idx) {
    if (get_global_id(0) == 0) data[idx] = 0;
}
"""

def demo_blelloch_scan(ctx, queue):
    section("1", "BLELLOCH (WORK-EFFICIENT) SCAN",
            "O(N) work, O(log N) steps — exclusive prefix sum")

    print("""
  Hillis-Steele vs Blelloch:
  ──────────────────────────
  Hillis-Steele: inclusive, O(N log N) work, O(log N) steps
  Blelloch:      exclusive, O(N) work,      O(2 log N) steps

  Blelloch is WORK-EFFICIENT — does the minimum total operations.
  Better for large arrays where total work matters.

  Two phases:
  ───────────
  UP-SWEEP (reduce tree):      [1 2 3 4 5 6 7 8]
    stride=1: add pairs         [1 3 3 7 5 11 7 15]  ← unaffected
    stride=2: add pairs         [1 3 3 10 5 11 7 36] ← partial sums
    stride=4: add pairs         [1 3 3 10 5 11 7 36]

  Set last = 0:                 [1 3 3 10 5 11 7  0]

  DOWN-SWEEP (distribute):
    stride=4: swap+add          [1 3 3  0 5 11 7 10]
    stride=2: swap+add          [1 0 3  3 5 10 7 21]
    stride=1: swap+add          [0 1 3  6 10 15 21 28]
                                 ↑ exclusive scan result!
""")

    # Power of 2 for simplicity
    N = 256
    data = np.arange(1, N + 1, dtype=np.int32)  # [1, 2, 3, ..., 256]

    mf = cl.mem_flags
    prg = cl.Program(ctx, BLELLOCH_KERNEL).build()

    # Copy data to device
    d_data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data.copy())

    # Up-sweep phase
    stride = 1
    while stride < N:
        num_items = N // (stride * 2)
        if num_items > 0:
            prg.upsweep(queue, (num_items,), None, d_data,
                        np.int32(stride), np.int32(N)).wait()
        stride *= 2

    # Set last element to 0
    prg.set_zero(queue, (1,), None, d_data, np.int32(N - 1)).wait()

    # Down-sweep phase
    stride = N // 2
    while stride >= 1:
        num_items = N // (stride * 2)
        if num_items > 0:
            prg.downsweep(queue, (num_items,), None, d_data,
                          np.int32(stride), np.int32(N)).wait()
        stride //= 2

    # Read back result
    result = np.empty(N, dtype=np.int32)
    cl.enqueue_copy(queue, result, d_data).wait()

    # CPU reference (exclusive scan)
    cpu_result = np.zeros(N, dtype=np.int32)
    cpu_result[1:] = np.cumsum(data[:-1])

    match = np.array_equal(result, cpu_result)
    print(f"  N = {N}")
    print(f"  Input[:8]:   {data[:8]}")
    print(f"  GPU[:8]:     {result[:8]}")
    print(f"  CPU[:8]:     {cpu_result[:8]}")
    print(f"  GPU[-4:]:    {result[-4:]}")
    print(f"  CPU[-4:]:    {cpu_result[-4:]}")
    print(f"  Result:      {'PASS' if match else 'FAIL'}")

    # Count operations
    total_work_hs = N * int(np.log2(N))  # Hillis-Steele
    total_work_bl = 2 * N  # Blelloch (up + down)
    print(f"\n  Work comparison for N={N}:")
    print(f"    Hillis-Steele: {total_work_hs:,} operations")
    print(f"    Blelloch:      {total_work_bl:,} operations")
    print(f"    Ratio:         {total_work_hs / total_work_bl:.1f}x more work for Hillis-Steele")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • Exclusive scan: output[i] = sum(input[0..i-1]), output[0] = 0
  • Inclusive scan: output[i] = sum(input[0..i])
  • Convert exclusive → inclusive: inclusive[i] = exclusive[i] + input[i]
  • Blelloch is preferred for LARGE arrays (less total work)
  • Hillis-Steele is simpler and faster for SMALL arrays (fewer steps)
  • Both are O(log N) steps — same parallel depth
""")


# ===================================================================
# 2. MULTI-BLOCK SCAN — Scanning Millions of Elements
# ===================================================================
# Single work-group scan is limited to ~256-1024 elements.
# For large arrays: scan-then-propagate (3-kernel approach).
#
#   1. Per-block scan: each work-group scans its chunk
#   2. Scan the block sums
#   3. Add block sums to each element
# ===================================================================

MULTIBLOCK_KERNEL = """
// Per-block inclusive scan using local memory (Hillis-Steele)
__kernel void block_scan(__global const int* input,
                         __global int* output,
                         __global int* block_sums,
                         __local int* temp,
                         int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gs = get_local_size(0);
    int block_id = get_group_id(0);

    // Load into local memory
    temp[lid] = (gid < N) ? input[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Hillis-Steele inclusive scan within work-group
    for (int stride = 1; stride < gs; stride <<= 1) {
        int val = (lid >= stride) ? temp[lid - stride] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        temp[lid] += val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write scanned result
    if (gid < N) output[gid] = temp[lid];

    // Last item in block writes block total
    if (lid == gs - 1) {
        block_sums[block_id] = temp[lid];
    }
}

// Add block prefix to each element (to complete the global scan)
__kernel void add_block_sums(__global int* data,
                             __global const int* block_prefixes,
                             int N) {
    int gid = get_global_id(0);
    int block_id = get_group_id(0);
    if (block_id > 0 && gid < N) {
        data[gid] += block_prefixes[block_id - 1];
    }
}
"""

def demo_multiblock_scan(ctx, queue):
    section("2", "MULTI-BLOCK SCAN",
            "Scan millions of elements across many work-groups")

    N = 1 << 16  # 65,536 elements
    LS = 256
    NUM_BLOCKS = (N + LS - 1) // LS
    data = np.ones(N, dtype=np.int32)  # All 1s → scan = [1, 2, 3, ..., N]

    mf = cl.mem_flags
    prg = cl.Program(ctx, MULTIBLOCK_KERNEL).build()

    d_input = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    d_output = cl.Buffer(ctx, mf.READ_WRITE, data.nbytes)
    d_block_sums = cl.Buffer(ctx, mf.READ_WRITE, NUM_BLOCKS * 4)

    print(f"""
  Three-kernel approach for large arrays:
  ────────────────────────────────────────
  1. block_scan:      each WG scans its chunk independently
  2. scan block_sums: prefix sum of per-block totals (recursive!)
  3. add_block_sums:  propagate prefix to all elements

  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Block 0 │ │ Block 1 │ │ Block 2 │ │ Block 3 │  ← Step 1: per-block scan
  │ [1..256]│ │[1..256] │ │[1..256] │ │[1..256] │
  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
       │           │           │           │
    sum=256     sum=256     sum=256     sum=256
       │           │           │           │       ← Step 2: scan block sums
       └→ [256,    512,       768,       1024]
                    │           │           │      ← Step 3: add prefix
            +256 to   +512 to    +768 to
            block 1   block 2    block 3

  N = {N:,}, Blocks = {NUM_BLOCKS}, Block size = {LS}
""")

    # Step 1: Per-block scan
    evt = prg.block_scan(queue, (NUM_BLOCKS * LS,), (LS,),
                         d_input, d_output, d_block_sums,
                         cl.LocalMemory(LS * 4), np.int32(N))
    evt.wait()

    # Step 2: Scan block sums on CPU (for simplicity; could recurse on GPU)
    block_sums = np.empty(NUM_BLOCKS, dtype=np.int32)
    cl.enqueue_copy(queue, block_sums, d_block_sums).wait()
    block_prefixes = np.cumsum(block_sums).astype(np.int32)
    d_block_prefixes = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                 hostbuf=block_prefixes)

    # Step 3: Add block prefixes
    prg.add_block_sums(queue, (NUM_BLOCKS * LS,), (LS,),
                       d_output, d_block_prefixes, np.int32(N)).wait()

    # Read result
    result = np.empty(N, dtype=np.int32)
    cl.enqueue_copy(queue, result, d_output).wait()

    # CPU reference
    cpu_result = np.cumsum(data).astype(np.int32)

    match = np.array_equal(result, cpu_result)
    print(f"  Result[:8]:     {result[:8]}")
    print(f"  Expected[:8]:   {cpu_result[:8]}")
    print(f"  Result[255:258]: {result[255:258]}  (block boundary)")
    print(f"  Result[-4:]:    {result[-4:]}")
    print(f"  Expected[-1]:   {cpu_result[-1]}")
    print(f"  Result:         {'PASS' if match else 'FAIL'}")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • This 3-kernel pattern scales to billions of elements
  • Step 2 can be recursive (scan-of-scans) for huge arrays
  • Total work: O(N), total steps: O(log N * num_levels)
  • Same pattern used in: radix sort, stream compaction, BFS
  • In practice: use library (CUB, rocPRIM) — but know the algorithm
""")


# ===================================================================
# 3. BITONIC SORT — Data-Oblivious Sorting Network
# ===================================================================
# A comparison network that works regardless of input order.
# Perfect for GPU: fixed comparison pattern, no data-dependent branching.
#
#   O(N log^2 N) comparisons, O(log^2 N) parallel steps
# ===================================================================

BITONIC_KERNEL = """
__kernel void bitonic_step(__global int* data,
                           int j, int k, int N) {
    int i = get_global_id(0);
    if (i >= N) return;

    int ij = i ^ j;  // Partner element

    if (ij > i) {    // Only one thread per pair does the swap
        bool ascending = ((i & k) == 0);

        if (ascending) {
            if (data[i] > data[ij]) {
                int tmp = data[i];
                data[i] = data[ij];
                data[ij] = tmp;
            }
        } else {
            if (data[i] < data[ij]) {
                int tmp = data[i];
                data[i] = data[ij];
                data[ij] = tmp;
            }
        }
    }
}
"""

def demo_bitonic_sort(ctx, queue):
    section("3", "BITONIC SORT",
            "Data-oblivious sorting network — O(N log^2 N)")

    # Must be power of 2
    N = 1024
    data = np.random.randint(0, 10000, N).astype(np.int32)

    print(f"""
  Bitonic sort: a COMPARISON NETWORK
  ───────────────────────────────────
  • Fixed pattern of compare-and-swap operations
  • Does NOT depend on input data → no branch divergence!
  • Perfect for GPU: all threads follow the same path

  Algorithm (for N elements):
    for k = 2, 4, 8, ..., N:        ← O(log N) stages
      for j = k/2, k/4, ..., 1:     ← O(log N) steps per stage
        compare-and-swap pairs at distance j

  Total: O(log^2 N) parallel steps, O(N log^2 N) comparisons

  Visualization (8 elements):
  ┌──────────────────────────────────────────┐
  │  [5 3 8 1 4 7 2 6]  Input               │
  │   ↕ ↕   ↕ ↕         k=2: pairs of 2     │
  │  [3 5 1 8 4 7 2 6]                       │
  │   ↕   ↕ ↕   ↕       k=4: merge to 4     │
  │  [1 3 5 8 2 4 6 7]                       │
  │   ↕       ↕ ↕       ↕  k=8: merge to 8  │
  │  [1 2 3 4 5 6 7 8]  Sorted!             │
  └──────────────────────────────────────────┘
""")

    mf = cl.mem_flags
    prg = cl.Program(ctx, BITONIC_KERNEL).build()
    d_data = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data.copy())

    # Bitonic sort: log^2(N) kernel launches
    t0 = time.perf_counter()
    k = 2
    while k <= N:
        j = k // 2
        while j >= 1:
            prg.bitonic_step(queue, (N,), None, d_data,
                             np.int32(j), np.int32(k), np.int32(N)).wait()
            j //= 2
        k *= 2
    gpu_ms = (time.perf_counter() - t0) * 1000

    # Read result
    result = np.empty(N, dtype=np.int32)
    cl.enqueue_copy(queue, result, d_data).wait()

    # CPU reference
    cpu_sorted = np.sort(data)

    match = np.array_equal(result, cpu_sorted)
    num_steps = sum(1 for k in _log2_range(N) for j in _log2_range_down(k))

    print(f"  N = {N}")
    print(f"  Input[:10]:  {data[:10]}")
    print(f"  GPU[:10]:    {result[:10]}")
    print(f"  CPU[:10]:    {cpu_sorted[:10]}")
    print(f"  Kernel launches: {num_steps}")
    print(f"  GPU time:    {gpu_ms:.2f} ms")
    print(f"  Result:      {'PASS' if match else 'FAIL'}")

    print("""
  INTERVIEW KEY POINTS:
  ─────────────────────
  • Data-oblivious: same comparisons regardless of input → no divergence
  • Compare pattern: i XOR j gives partner index
  • Direction: (i & k) == 0 → ascending, else descending
  • Must be power-of-2 (pad with INT_MAX for arbitrary N)
  • For GPU: each step is one kernel launch, all threads active
  • Alternative: Radix sort is faster in practice (O(N*W), W=word bits)
    but radix sort requires scan as a subroutine
  • Use in interviews to show you understand sorting networks
""")


def _log2_range(n):
    """Yield 2, 4, 8, ..., n."""
    k = 2
    while k <= n:
        yield k
        k *= 2

def _log2_range_down(k):
    """Yield k//2, k//4, ..., 1."""
    j = k // 2
    while j >= 1:
        yield j
        j //= 2


# ===================================================================
# 4. PRODUCER-CONSUMER PATTERN
# ===================================================================
# Classic concurrency pattern with a bounded buffer.
# Demonstrates: synchronization, condition variables, deadlock avoidance.
#
# GPU context: CPU produces data → GPU consumes (processes) → output
# ===================================================================

def demo_producer_consumer():
    section("4", "PRODUCER-CONSUMER PATTERN",
            "Bounded buffer with synchronization — classic concurrency")

    print("""
  ┌──────────────────────────────────────────────────────────────┐
  │  PRODUCER-CONSUMER with Bounded Buffer                       │
  │                                                              │
  │  Producer ──→ [ B U F F E R ] ──→ Consumer                   │
  │  (CPU)         (bounded, N)        (GPU / worker)            │
  │                                                              │
  │  If buffer FULL:  producer WAITS (backpressure)              │
  │  If buffer EMPTY: consumer WAITS (no work to do)             │
  │                                                              │
  │  Synchronization: mutex + condition variables (or queue)      │
  │  Deadlock risk: if producer waits for consumer AND vice versa │
  └──────────────────────────────────────────────────────────────┘
""")

    BUFFER_SIZE = 5
    NUM_ITEMS = 20
    buffer = stdlib_queue.Queue(maxsize=BUFFER_SIZE)
    produced = []
    consumed = []
    lock = threading.Lock()

    def producer():
        for i in range(NUM_ITEMS):
            item = f"item-{i}"
            buffer.put(item)  # Blocks if buffer full
            with lock:
                produced.append(item)
        buffer.put(None)  # Sentinel to signal done

    def consumer():
        while True:
            item = buffer.get()  # Blocks if buffer empty
            if item is None:
                break
            # Simulate processing
            result = item.upper()
            with lock:
                consumed.append(result)

    t0 = time.perf_counter()
    prod_thread = threading.Thread(target=producer)
    cons_thread = threading.Thread(target=consumer)
    prod_thread.start()
    cons_thread.start()
    prod_thread.join()
    cons_thread.join()
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"  Buffer size:    {BUFFER_SIZE}")
    print(f"  Items produced: {len(produced)}")
    print(f"  Items consumed: {len(consumed)}")
    print(f"  Time:           {elapsed:.2f} ms")
    print(f"  First 5 consumed: {consumed[:5]}")
    print(f"  Result:         {'PASS' if len(consumed) == NUM_ITEMS else 'FAIL'}")

    print("""
  GPU application of Producer-Consumer:
  ──────────────────────────────────────
  • Producer (CPU thread): reads frames / sensor data / batches
  • Buffer (host memory): ring buffer of prepared data
  • Consumer (GPU): processes each batch asynchronously
  • This IS the double-buffering pattern from Session 6!

  Deadlock avoidance rules:
  ─────────────────────────
  1. Always acquire locks in the SAME ORDER
  2. Use timeouts on blocking operations
  3. Use sentinel values (None) to signal completion
  4. Prefer lock-free queues for high throughput
  5. Python's queue.Queue is thread-safe (uses mutex internally)

  INTERVIEW KEY POINTS:
  ─────────────────────
  • Know the three components: producer, buffer, consumer
  • Buffer bound prevents unbounded memory growth
  • Thread-safe queue = mutex + 2 condition variables
    (not_full for producer, not_empty for consumer)
  • In GPU context: this is the streaming pipeline architecture
""")


# ===================================================================
# 5. CONCURRENCY HAZARDS
# ===================================================================
# The bugs that parallel programs uniquely suffer from.
# Interview MUST-KNOWS: deadlock, false sharing, CAS.
# ===================================================================

def demo_concurrency_hazards():
    section("5", "CONCURRENCY HAZARDS",
            "Deadlock, false sharing, CAS — the parallel programming bugs")

    # --- 5a: DEADLOCK ---
    print("""
  ┌──────────────────────────────────────────────────────────┐
  │  5a. DEADLOCK                                            │
  │  Two threads each hold a lock the other needs.           │
  │                                                          │
  │  Thread A: lock(X) → wants lock(Y) → BLOCKED            │
  │  Thread B: lock(Y) → wants lock(X) → BLOCKED            │
  │                                                          │
  │  Neither can proceed → DEADLOCK!                         │
  └──────────────────────────────────────────────────────────┘

  Four necessary conditions (Coffman conditions):
    1. Mutual exclusion   — resources held exclusively
    2. Hold and wait      — hold one, request another
    3. No preemption      — can't forcibly take a lock
    4. Circular wait      — A→B→C→...→A

  Break ANY one to prevent deadlock:
    • Lock ordering: always acquire locks in alphabetical/numeric order
    • Try-lock with timeout: give up if can't acquire within N ms
    • Single-lock: use one coarser lock instead of multiple fine ones

  GPU note: work-groups can deadlock if they try to synchronize
  with each other via global memory. GPU provides NO cross-WG barrier!
  Only barrier() within a work-group is safe.
""")

    # Demo: deadlock-SAFE version using lock ordering
    lock_A = threading.Lock()
    lock_B = threading.Lock()
    results = []

    def safe_thread_1():
        # Always acquire in order: A then B
        with lock_A:
            with lock_B:
                results.append("T1 done")

    def safe_thread_2():
        # Same order: A then B (NOT B then A!)
        with lock_A:
            with lock_B:
                results.append("T2 done")

    t1 = threading.Thread(target=safe_thread_1)
    t2 = threading.Thread(target=safe_thread_2)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print(f"  Deadlock-safe demo: {results}")

    # --- 5b: FALSE SHARING ---
    print("""
  ┌──────────────────────────────────────────────────────────┐
  │  5b. FALSE SHARING                                       │
  │  Different threads write to different variables that      │
  │  happen to be on the SAME cache line (64 bytes).          │
  │                                                          │
  │  Cache line: [counter_A | counter_B | ... ]              │
  │                                                          │
  │  Thread A writes counter_A → invalidates B's cache line  │
  │  Thread B writes counter_B → invalidates A's cache line  │
  │  Ping-pong! Massive slowdown despite NO logical sharing. │
  └──────────────────────────────────────────────────────────┘
""")

    # Demonstrate false sharing vs padded
    N_ITERS = 2_000_000

    # False sharing: array of adjacent ints
    shared_array = np.zeros(8, dtype=np.int64)
    # Padded: each counter on its own cache line (16 int64s = 128 bytes apart)
    padded_array = np.zeros(8 * 16, dtype=np.int64)

    def increment_shared(idx):
        for _ in range(N_ITERS):
            shared_array[idx] += 1

    def increment_padded(idx):
        for _ in range(N_ITERS):
            padded_array[idx * 16] += 1

    # Shared (false sharing likely)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=increment_shared, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    shared_ms = (time.perf_counter() - t0) * 1000

    # Padded (no false sharing)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=increment_padded, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    padded_ms = (time.perf_counter() - t0) * 1000

    print(f"  4 threads × {N_ITERS:,} increments:")
    print(f"    Adjacent (false sharing): {shared_ms:.1f} ms")
    print(f"    Padded (no false sharing): {padded_ms:.1f} ms")
    ratio = shared_ms / padded_ms if padded_ms > 0 else 0
    print(f"    Ratio: {ratio:.2f}x")
    print(f"    (Note: Python GIL limits true parallelism; in C/C++ the")
    print(f"     difference can be 10-50x on adjacent cache lines)")

    # --- 5c: COMPARE-AND-SWAP (CAS) ---
    print("""
  ┌──────────────────────────────────────────────────────────┐
  │  5c. COMPARE-AND-SWAP (CAS)                              │
  │  The atomic operation that enables LOCK-FREE programming. │
  │                                                          │
  │  CAS(address, expected, desired):                        │
  │    if (*address == expected):                             │
  │        *address = desired                                │
  │        return true                                       │
  │    else:                                                 │
  │        return false     ← someone else changed it!       │
  │                                                          │
  │  All done ATOMICALLY by hardware — no lock needed.       │
  └──────────────────────────────────────────────────────────┘

  Lock-free increment using CAS (pseudocode):
  ────────────────────────────────────────────
  do {
      old = *counter;
      new = old + 1;
  } while (!CAS(counter, old, new));  // Retry if someone beat us

  On GPU (OpenCL):
  ────────────────
  atomic_cmpxchg(ptr, expected, desired)  // Returns old value
  atomic_add(ptr, 1)  // Simpler — uses CAS internally

  When to use CAS vs locks:
  ─────────────────────────
  • CAS: low contention, simple operations (increment, min, max)
  • Locks: high contention, complex multi-step operations
  • CAS can livelock under extreme contention (all threads retry)
  • ABA problem: value changes A→B→A, CAS thinks nothing changed
    Fix: use version counter (tagged pointer)

  INTERVIEW KEY POINTS:
  ─────────────────────
  • CAS is the building block of ALL lock-free data structures
  • On GPU: atomic_add/atomic_min/atomic_max use CAS internally
  • GPU histogram uses atomic_add — CAS applied!
  • Know the retry loop pattern
  • Know the ABA problem and tagged-pointer fix
""")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  SESSION 7: ADVANCED ALGORITHMS & CONCURRENCY PATTERNS")
    print("  Filling the critical gaps")
    print("=" * 70)

    ctx, queue, device = get_gpu_context()
    print(f"  GPU: {device.name} ({device.max_compute_units} CUs)")

    demo_blelloch_scan(ctx, queue)
    demo_multiblock_scan(ctx, queue)
    demo_bitonic_sort(ctx, queue)
    demo_producer_consumer()
    demo_concurrency_hazards()

    print("\n" + "=" * 70)
    print("  SESSION 7 COMPLETE — ADVANCED ALGORITHMS SUMMARY")
    print("=" * 70)
    print("""
  Algorithms learned:
  ───────────────────
  1. Blelloch Scan:     Work-efficient O(N), exclusive prefix sum
  2. Multi-Block Scan:  3-kernel scan-then-propagate for large arrays
  3. Bitonic Sort:      Data-oblivious sorting network, O(N log^2 N)
  4. Producer-Consumer: Bounded buffer, thread-safe queue, sentinel
  5. Concurrency Hazards:
     a. Deadlock:       4 Coffman conditions, lock ordering fix
     b. False Sharing:  Cache-line ping-pong, padding fix
     c. CAS:            Lock-free atomic, retry loop, ABA problem

  Scan comparison:
  ────────────────
  Hillis-Steele: inclusive, O(N log N) work, simple
  Blelloch:      exclusive, O(N) work, complex (two phases)
  Multi-block:   any size, 3-kernel pattern, recursive

  Next: Session 8 — AMD-specific optimizations & NPU deep dive
""")


if __name__ == "__main__":
    main()
