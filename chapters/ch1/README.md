# Chapter 1 - Introduction to Deep Learning

tutorial: [chapter 1](https://dlvr.rantai.dev/docs/part-i/chapter-1/)

## Prerequisites

[`tch`](https://github.com/laurentmazare/tch-rs) depends on the following libraries:

- [libtorch](https://pytorch.org/)
- [CUDA toolkit](https://developer.nvidia.com/cuda/toolkit)

After installing libtorch, set the following environment variable:

```sh
export LIBTORCH=/opt/libtorch   # or other path where libtorch is
export LD_LIBRARY_PATH="${LIBTORCH}/lib":$LD_LIBRARY_PATH
export LIBTORCH_BYPASS_VERSION_CHECK="1"
```

If you're using Archlinux, you could install these libraries via AUR:

```sh
paru -S libtorch-cuda cuda
```

## Dependencies version

- `cuda` 13.3.0-1
- `libtorch` 2.12.0 (CUDA 13.2)
