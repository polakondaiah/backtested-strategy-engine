#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

// Hot loop: bar-by-bar PnL with transaction cost.
// position[i] * fwd_ret[i] - turnover[i]*cost
// This is the loop that dominates backtest time for large universes.

py::array_t<double> compute_pnl(
    py::array_t<double> position,
    py::array_t<double> fwd_ret,
    double cost_bps
) {
    auto pos = position.unchecked<1>();
    auto ret = fwd_ret.unchecked<1>();
    ssize_t n = pos.shape(0);
    auto result = py::array_t<double>(n);
    auto res = result.mutable_unchecked<1>();
    double cost = cost_bps / 10000.0;
    double prev_pos = 0.0;
    for (ssize_t i = 0; i < n; ++i) {
        double p = pos(i);
        double turnover = std::fabs(p - prev_pos);
        if (i == 0) turnover = std::fabs(p);
        res(i) = p * ret(i) - turnover * cost;
        prev_pos = p;
    }
    return result;
}

// Benchmark helper: run loop many times
double benchmark_loop(py::array_t<double> position, py::array_t<double> fwd_ret, double cost_bps, int iterations) {
    for (int k = 0; k < iterations; ++k) {
        auto r = compute_pnl(position, fwd_ret, cost_bps);
        (void)r;
    }
    return 0.0;
}

PYBIND11_MODULE(pnl_cpp, m) {
    m.doc() = "C++ PnL inner loop for backtest";
    m.def("compute_pnl", &compute_pnl, "Compute PnL with transaction costs",
          py::arg("position"), py::arg("fwd_ret"), py::arg("cost_bps"));
    m.def("benchmark_loop", &benchmark_loop, "Benchmark loop",
          py::arg("position"), py::arg("fwd_ret"), py::arg("cost_bps"), py::arg("iterations"));
}
