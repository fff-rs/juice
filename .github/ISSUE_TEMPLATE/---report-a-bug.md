---
name: "\U0001F41B Report a bug"
about: Create a report to help us fix bugs
title: ''
labels: bug
assignees: dev

---

**Describe the bug**

<!-- A clear and concise description of what the bug is.

Bad:
 > sh*t ain't workin, it's ALL BROKEN
 > cuda does not converge

Good:
 > Build failure with README instructions
 > Network fails to converge using the RNN layer with the cuda backend
-->

**To Reproduce**

Steps to reproduce the behaviour:

1. Setting up preconditions with `...`
2. Run `cargo ...`
3. Execute `...`
4. ...

**Expected behavior**

<!-- A clear and concise description of what you expected to happen. -->

**Screenshots**

<!-- If applicable, add complete copies of the commandline output to help explain your problem.
Use code blocks with ` ```plain ` or ` ```rust ` depending on the output type. -->

**Please complete the following information:**
<!-- assumes `ripgrep` (with the binary `rg`) is installed -->
 - System: <!-- Fedora, Ubuntu, Win10, MacOS, ... with a version -->
 - Version: <!-- git sha, dependency release version -->
 - Rust: <!-- $( rustc --version ) -->
 - Environment: <!-- $( printenv | rg '^(CU(BLAS|DNN)|(OPEN)?BLAS|(OPEN)?CL)_(VARIANT|STATIC|LIBS_(LIB|INCLUDE)_DIR)=.*$' )$ -->
 - Backends (if relevant):
   * opencl: <!-- $( clinfo -l ) -->
   * cuda: <!-- $( nvidia-smi ) -->

**Additional context**

<!-- Add any other relevant context here. -->
