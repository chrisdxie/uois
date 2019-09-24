from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hough_voting',
    ext_modules=[
        CUDAExtension(
            name='hough_voting_cuda', 
            sources = [
            'hough_voting_layer.cpp',
            'hough_voting_kernel.cu'],
            include_dirs = ['/usr/local/include/eigen3'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
