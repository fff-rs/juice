#! /bin/bash
if [ $# -eq 0 ]
  then
    echo "No benchmark name supplied"
    exit 1
fi
benchname=$1
mkdir -p target/perf
perf record -a -g --output target/perf/${benchname}.data target/debug/rblas_overhead-c02a41a1401d43da --bench ${benchname}
perf script -f -i target/perf/${benchname}.data > target/perf/${benchname}.scripted
stackcollapse-perf target/perf/${benchname}.scripted | grep ${benchname} > target/perf/${benchname}.folded
flamegraph target/perf/${benchname}.folded > target/perf/${benchname}.svg
