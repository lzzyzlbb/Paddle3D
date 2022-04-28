from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='iou3d_nms',
    ext_modules=CUDAExtension(sources=[
        'csrc/iou3d_cpu.cpp', 'csrc/iou3d_nms_api.cpp',
        'csrc/iou3d_nms_kernel.cu', 'csrc/iou3d_nms.cpp'
    ]))
