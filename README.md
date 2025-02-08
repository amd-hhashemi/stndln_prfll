# Cuda Prefill Boilerplate based on Flashinfer

# About

This boilerplate creates one program that runs one prefill kernel,
with tensors dumped from flashinfer indected as input after loading 
from disk, and final result can be compared with reference.
The idea is to have cuda code built and linked via nvcc and host/app
code built with g++. We're using the separate compilation mode with
nvcc, so the Makefile has an extra step that links the cuda code.
The host linker will link the object(s) created by nvcc.


# Requirements

Currently testing on a GTX 960 (Nvidia compute level 5.2).
Although, the compute value can be adjusted by `NVCC_FLAGS` in the Makefile.
GNU Make, g++.
You need graphics drivers for your Nvidia chipset, the Nvidia sdk, nvcc.


# Building

It works with GNU Make. Do a `make && ./build/program`.


# License

This project, `cuda-boilerplate` by Alexander Ressler,
is licensed under the MIT license.
