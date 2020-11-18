from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='pixel_nms',
      ext_modules=[cpp_extension.CppExtension('pixel_nms', ['pixel_nms_kernel.cu', 'pixel_nms.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
