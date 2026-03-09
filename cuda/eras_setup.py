from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='eras',
    ext_modules=[
        CUDAExtension(
            name='eras',
            sources=[
                'eras.cpp',
                'eras_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-lineinfo'
                ],
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
