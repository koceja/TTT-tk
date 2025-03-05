import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from config import sources, target, kernels
target = target.lower()

# Emulate the env.src settings (set environment variables)
os.environ["PYTHONPATH"] = os.path.join(os.getcwd(), "src/common/pyutils")
os.environ["THUNDERKITTENS_ROOT"] = os.getcwd()
os.environ["LIBTORCH_PATH"] = os.path.join(os.getcwd(), "libtorch")


# Set environment variables
thunderkittens_root = os.getenv('THUNDERKITTENS_ROOT', os.path.abspath(os.path.join(os.getcwd(), '.')))
python_include = subprocess.check_output(['python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()
torch_include = subprocess.check_output(['python', '-c', "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"]).decode().strip()
# print('Thunderkittens root:', thunderkittens_root)
# print('Python include:', python_include)
# print('Torch include directories:', torch_include)

# CUDA flags
cuda_flags = [
    '-DNDEBUG',
    '-Xcompiler=-Wno-psabi',
    '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler',
    '--use_fast_math',
    '-std=c++20',
    '-O3',
    '-Xnvlink=--verbose',
    '-Xptxas=--verbose',
    '-Xptxas=--warn-on-spills',
    # '-G',
    '-diag-suppress=3189', # "module is parsed as an identifier rather than a keyword"
    f'-I{thunderkittens_root}/include',
    f'-I{thunderkittens_root}/prototype',
    f'-I{python_include}',
    '-DTORCH_COMPILE'
] + torch_include.split()
cpp_flags = [
    '-std=c++20',
    '-O3'
]

if target == '4090':
    cuda_flags.append('-DKITTENS_4090')
    cuda_flags.append('-arch=sm_89')
elif target == 'h100':
    cuda_flags.append('-DKITTENS_HOPPER')
    cuda_flags.append('-arch=sm_90a')
elif target == 'a100':
    cuda_flags.append('-DKITTENS_A100')
    cuda_flags.append('-arch=sm_80')
else:
    raise ValueError(f'Target {target} not supported')

source_files = ['thunderkittens.cpp']
for k in kernels:
    if target not in sources[k]['source_files']:
        raise KeyError(f'Target {target} not found in source files for kernel {k}')
    if type(sources[k]['source_files'][target]) == list:
        source_files.extend(sources[k]['source_files'][target])
    else:
        source_files.append(sources[k]['source_files'][target])
    cpp_flags.append(f'-DTK_COMPILE_{k.replace(" ", "_").upper()}')

setup(
    name='test_time_training',
    version='0.10.0',
    author='Daniel Koceja',
    author_email='dankoceja@gmail.com',
    description='A library for test-time training.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    url='https://github.com/koceja/TTT-tk',
    install_requires=[
        'torch>=2.4.0',
    ],
    ext_modules=[
        CUDAExtension(
            'test_time_training',
            sources=source_files, 
            extra_compile_args={'cxx' : cpp_flags,
                                'nvcc' : cuda_flags}, 
            libraries=['cuda']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)