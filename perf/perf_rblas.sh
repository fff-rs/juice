#! /bin/bash
perf record -a -g --output perf_rblas_data.perf target/debug/rblas_overhead-cf1a2670c118749d --bench bench_1000_dot_100_rblas
perf script -f -i perf_rblas_data.perf > perf_rblas_script.perf
/home/hobofan/stuff/FlameGraph/stackcollapse-perf.pl perf_rblas_script.perf > perf_rblas_folded.perf
/home/hobofan/stuff/FlameGraph/flamegraph.pl perf_rblas_folded.perf > perf_rblas_graph.svg
