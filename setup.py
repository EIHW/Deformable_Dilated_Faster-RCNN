#!/usr/bin/python

import glob
import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension


def search_dir(root, name):
    for sub_root, d_names, f_names in os.walk(os.path.join(root, 'src')):
        if name.lower() in sub_root.lower():
            return sub_root
    return None


def get_extensions(name):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = search_dir(this_dir, name)
    if extensions_dir is None:
        raise FileNotFoundError('Could not find: {}'.format(name))

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    os.environ["CC"] = "g++"
    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}  # '/MD'
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        # raise NotImplementedError('Cuda is not available')
        pass

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "_ext",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


with open('requirements.txt') as req:
    required = req.read().splitlines()

setup(
    name='Deformable Dilated Faster-RCNN for Universal Lesion Detection in CT Images',
    version='1.0.0',
    author='Fabio Hellmann',
    author_email='fabio.hellmann@informatik.uni-augsburg.de',
    packages=find_packages(),
    data_files=[('', ['__main__.py', ])],
    ext_modules=get_extensions('deform_v2'),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    test_suite='tests',
    install_requires=required
)
