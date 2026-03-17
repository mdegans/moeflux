"""Build script for fast_expert_io, fast_weight_load, and fast_moe_load C extensions."""

from setuptools import setup, Extension
import numpy as np

fast_expert_io = Extension(
    'fast_expert_io',
    sources=['fast_expert_io.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        '-O3',
        '-march=armv8.4-a',
        '-flto',
        '-DPAGE_SIZE=16384',
        '-Wno-deprecated-declarations',
    ],
    extra_link_args=[
        '-lpthread',
        '-flto',
    ],
)

fast_weight_load = Extension(
    'fast_weight_load',
    sources=['fast_weight_load.c'],
    extra_compile_args=[
        '-O3',
        '-march=armv8.4-a',
        '-flto',
        '-Wno-deprecated-declarations',
    ],
    extra_link_args=[
        '-lpthread',
        '-flto',
    ],
)

fast_moe_load = Extension(
    'fast_moe_load',
    sources=['fast_moe_load.c'],
    extra_compile_args=[
        '-O3',
        '-march=armv8.4-a',
        '-flto',
        '-Wno-deprecated-declarations',
    ],
    extra_link_args=[
        '-lpthread',
        '-flto',
    ],
)

setup(
    name='ane_research_cext',
    version='0.3.0',
    description='High-throughput expert weight I/O C extensions',
    ext_modules=[fast_expert_io, fast_weight_load, fast_moe_load],
)
