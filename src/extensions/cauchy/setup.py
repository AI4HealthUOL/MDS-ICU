from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import torch

# Ensure CUDA_HOME is set
CUDA_HOME = os.environ.get('CUDA_HOME', None)
if CUDA_HOME is None:
    raise EnvironmentError("CUDA_HOME environment variable not set. Please set it to your CUDA path.")

# Define the CUDA extension
ext_modules = [
    CUDAExtension(
        name='cauchy_mult',
        sources=['cauchy.cpp', 'cauchy_cuda.cu'],
        extra_compile_args={
            'cxx': ['-O3', '-g', '-march=native', '-funroll-loops', '-std=c++17'],
            'nvcc': [
                '-O2',
                '-lineinfo',
                '--use_fast_math',
                '-arch=sm_80',  # adjust according to your GPU compute capability
            ],
        },
    )
]

setup(
    name='cauchy_mult',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},  # disable ninja to avoid fallback warning
)

