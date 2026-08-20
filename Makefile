.PHONY: data backtest cpp benchmark pipeline clean
data:
	python3 fetch_data.py
backtest:
	python3 -m src.backtest
cpp:
	c++ -O3 -Wall -shared -std=c++17 -fPIC $$(python3 -m pybind11 --includes) src/cpp/pnl.cpp -o pnl_cpp$$(python3-config --extension-suffix)
benchmark: cpp
	python3 benchmark.py
pipeline:
	bash run_pipeline.sh
clean:
	rm -rf results __pycache__ src/__pycache__ pnl_cpp*.so pipeline.log
