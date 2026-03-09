# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hulk_boost_rationals',
    ext_modules=[
        CUDAExtension(
            name='hulk_boost_rationals',
            sources=[
                'hulk_boost_rationals.cpp',
                'hulk_boost_rationals_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xcompiler', '-fPIC',
                    # Include Link-Time Optimization if desired
                    # '-dlto',
                    # Target multiple architectures for compatibility
                    '-gencode=arch=compute_70,code=sm_70',   # Volta GPUs
                    '-gencode=arch=compute_75,code=sm_75',   # Turing GPUs
                    '-gencode=arch=compute_80,code=sm_80',   # Ampere GPUs
                    '-gencode=arch=compute_86,code=sm_86',   # Ampere RTX30 Series
                    '-gencode=arch=compute_89,code=sm_89',   # Hopper GPUs
                    '-gencode=arch=compute_90,code=sm_90',   # Ada Lovelace GPUs
                    '-gencode=arch=compute_90,code=compute_90',  # PTX for future GPUs
                ],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
