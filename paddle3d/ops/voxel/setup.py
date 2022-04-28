import paddle
from paddle.utils.cpp_extension import CUDAExtension, CppExtension, setup

if paddle.device.is_compiled_with_cuda():
    setup(
        name='paddle3d.ops.voxelize',
        ext_modules=CUDAExtension(
            sources=['csrc/voxelize_op.cc', 'csrc/voxelize_op.cu'],
            extra_compile_args={'cxx': ['-DPADDLE_WITH_CUDA']}))
else:
    setup(
        name='paddle3d.ops.voxelize',
        ext_modules=CppExtension(sources=['csrc/voxelize_op.cc']))
