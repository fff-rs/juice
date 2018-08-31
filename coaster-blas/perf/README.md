# Profiling

Collenchyma comes with scripts to help with profiling performance problems.

Run [perf](http://www.brendangregg.com/perf.html) on one of the benchmark test:

```sh
# compile latest version of benchmarks with DWARF information
cargo rustc --bench rblas_overhead -- -g
sudo ./perf/run_perf.sh bench_1000_dot_100_collenchyma # perf needs sudo
```
