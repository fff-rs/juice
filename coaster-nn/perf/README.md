# Profiling

Collenchyma comes with scripts to help with profiling performance problems.

Run [perf](http://www.brendangregg.com/perf.html) on one of the benchmark test:

```sh
# compile latest version of benchmarks with DWARF information
cargo rustc --bench [bench_file_name] -- -g
sudo ./perf/run_perf.sh [bench_fn_name] # perf needs sudo
```
