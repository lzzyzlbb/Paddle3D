from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='grid_sample_3d',
    ext_modules=CUDAExtension(sources=[
        'grid_sample_3d.cc', 'grid_sample_3d.cu',
    ]))
