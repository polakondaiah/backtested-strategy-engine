"""Benchmark Python loop vs C++ pybind11 loop."""
import time
import numpy as np

COST_BPS = 5

def python_pnl(position, fwd_ret, cost_bps):
    cost = cost_bps / 10000.0
    n = len(position)
    res = np.empty(n)
    prev = 0.0
    for i in range(n):
        p = position[i]
        turnover = abs(p - prev) if i != 0 else abs(p)
        res[i] = p * fwd_ret[i] - turnover * cost
        prev = p
    return res

def main():
    n = 1_000_000
    rng = np.random.default_rng(0)
    pos = rng.uniform(-1, 1, n)
    fwd = rng.normal(0, 0.01, n)

    # warmup
    python_pnl(pos[:1000], fwd[:1000], COST_BPS)

    # Python timing
    t0 = time.perf_counter()
    for _ in range(5):
        python_pnl(pos, fwd, COST_BPS)
    t_py = (time.perf_counter() - t0) / 5

    # C++ timing
    try:
        import pnl_cpp
        t0 = time.perf_counter()
        for _ in range(5):
            pnl_cpp.compute_pnl(pos, fwd, COST_BPS)
        t_cpp = (time.perf_counter() - t0) / 5
        speedup = t_py / t_cpp if t_cpp > 0 else float('inf')
        # verify correctness
        py_res = python_pnl(pos[:1000], fwd[:1000], COST_BPS)
        cpp_res = pnl_cpp.compute_pnl(pos[:1000], fwd[:1000], COST_BPS)
        max_err = np.max(np.abs(py_res - cpp_res))
        print(f"Python avg: {t_py*1000:.2f} ms")
        print(f"C++    avg: {t_cpp*1000:.2f} ms")
        print(f"Speedup: {speedup:.1f}x")
        print(f"Max error (1000): {max_err:.2e}")
        # write result for resume
        with open("results/benchmark.txt","w") as f:
            f.write(f"python_ms: {t_py*1000:.2f}\ncpp_ms: {t_cpp*1000:.2f}\nspeedup: {speedup:.1f}x\n")
    except ImportError as e:
        print(f"C++ module not built ({e}) - Python only benchmark: {t_py*1000:.2f} ms")
        print("Build with: c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) src/cpp/pnl.cpp -o pnl_cpp$(python3-config --extension-suffix)")
        with open("results/benchmark.txt","w") as f:
            f.write(f"python_ms: {t_py*1000:.2f}\ncpp_not_built: {e}\n")

if __name__ == "__main__":
    import pathlib
    pathlib.Path("results").mkdir(exist_ok=True)
    main()
