#include "voxelize_op.h"
#include "paddle/extension.h"

template <typename T, typename T_int>
void dynamic_voxelize_cpu_kernel(const T *points, T_int *coors,
                                 const std::vector<float> voxel_size,
                                 const std::vector<float> coors_range,
                                 const std::vector<int> grid_size,
                                 const int64_t num_points,
                                 const int num_features, const int NDim) {
  const int ndim_minus_1 = NDim - 1;
  bool failed = false;
  // int coor[NDim];
  int *coor = new int[NDim]();
  int c;

  for (int64_t i = 0; i < num_points; ++i) {
    failed = false;
    int64_t offset = num_features * i;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points[offset + j] - coors_range[j]) / voxel_size[j]);
      // necessary to rm points out of range
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }

    offset = 3 * i;
    for (int k = 0; k < NDim; ++k) {
      if (failed) {
        coors[offset + k] = -1;
      } else {
        coors[offset + k] = coor[k];
      }
    }
  }

  delete[] coor;
  return;
}

template <typename T, typename T_int>
void hard_voxelize_cpu_kernel(
    const T *points, T *voxels, T_int *coors, T_int *num_points_per_voxel,
    T_int *coor_to_voxelidx, T_int *voxel_num,
    const std::vector<float> voxel_size, const std::vector<float> coors_range,
    const std::vector<int> grid_size, const int max_points,
    const int max_voxels, const int64_t num_points, const int num_features,
    const int NDim, const std::vector<int64_t> voxels_shape,
    const std::vector<int64_t> coors_shape,
    const std::vector<int64_t> num_points_per_voxel_shape,
    const std::vector<int64_t> coor_to_voxelidx_shape) {
  std::fill(voxels, voxels + max_voxels * max_points * num_features,
            static_cast<T>(0));
  // declare a temp coors & fill with 0
  auto temp_coors = paddle::Tensor(paddle::PlaceType::kCPU, {num_points, NDim});
  auto *coor = temp_coors.mutable_data<T_int>();
  std::fill(coor, coor + temp_coors.size(), static_cast<T_int>(0));

  // First use dynamic voxelization to get coors,
  // then check max points/voxels constraints
  dynamic_voxelize_cpu_kernel<T, T_int>(points, coor, voxel_size, coors_range,
                                        grid_size, num_points, num_features,
                                        NDim);
  int voxelidx, num;

  for (int64_t i = 0; i < num_points; ++i) {
    if (coor[i * NDim] == -1) {
      continue;
    }
    voxelidx = coor_to_voxelidx[coor[i * NDim] * coor_to_voxelidx_shape[1] *
                                    coor_to_voxelidx_shape[2] +
                                coor[i * NDim + 1] * coor_to_voxelidx_shape[2] +
                                coor[i * NDim + 2]];
    // record voxel
    if (voxelidx == -1) {
      voxelidx = voxel_num[0];
      if (max_voxels != -1 && voxel_num[0] >= max_voxels) {
        continue;
      }
      voxel_num[0] += 1;
      coor_to_voxelidx[coor[i * NDim] * coor_to_voxelidx_shape[1] *
                           coor_to_voxelidx_shape[2] +
                       coor[i * NDim + 1] * coor_to_voxelidx_shape[2] +
                       coor[i * NDim + 2]] = voxelidx;

      for (int k = 0; k < NDim; ++k) {
        coors[voxelidx * coors_shape[1] + k] = coor[i * NDim + k];
      }
    }

    // put points into voxel
    num = num_points_per_voxel[voxelidx];
    if (max_points == -1 || num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels[voxelidx * voxels_shape[1] * voxels_shape[2] +
               num * voxels_shape[2] + k] = points[i * num_features + k];
      }
      num_points_per_voxel[voxelidx] += 1;
    }
  }

  return;
}

#define CHECK_INPUT_CPU(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> dynamic_voxelize_cpu(
    const paddle::Tensor &points, const std::vector<float> &voxel_size,
    const std::vector<float> &coors_range, const int NDim = 3) {
  // check device
  CHECK_INPUT_CPU(points);

  std::vector<int> grid_size(NDim);
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  auto num_points = points.shape()[0];
  auto num_features = points.shape()[1];

  auto coors = paddle::Tensor(paddle::PlaceType::kCPU, {num_points, 3});
  auto *coors_data = coors.mutable_data<int>();
  std::fill(coors_data, coors_data + coors.size(), static_cast<int>(0));

  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "dynamic_voxelize_cpu_kernel", ([&] {
        dynamic_voxelize_cpu_kernel<data_t, int>(
            points.data<data_t>(), coors_data, voxel_size, coors_range,
            grid_size, num_points, num_features, NDim);
      }));

  return {coors};
}

std::vector<paddle::Tensor> hard_voxelize_cpu(
    const paddle::Tensor &points, const std::vector<float> &voxel_size,
    const std::vector<float> &coors_range, int max_points, int max_voxels,
    int NDim = 3) {
  // check device
  CHECK_INPUT_CPU(points);

  std::vector<int> grid_size(NDim);
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  auto num_points = points.shape()[0];
  auto num_features = points.shape()[1];

  auto voxels = paddle::Tensor(paddle::PlaceType::kCPU,
                               {max_voxels, max_points, num_features});

  auto coors = paddle::Tensor(paddle::PlaceType::kCPU, {max_voxels, 3});
  auto *coors_data = coors.mutable_data<int>();
  std::fill(coors_data, coors_data + coors.size(), static_cast<int>(0));

  auto num_points_per_voxel =
      paddle::Tensor(paddle::PlaceType::kCPU, {max_voxels});
  auto *num_points_per_voxel_data = num_points_per_voxel.mutable_data<int>();
  std::fill(num_points_per_voxel_data,
            num_points_per_voxel_data + num_points_per_voxel.size(),
            static_cast<int>(0));

  auto coor_to_voxelidx = paddle::Tensor(
      paddle::PlaceType::kCPU, {grid_size[2], grid_size[1], grid_size[0]});
  auto *coor_to_voxelidx_data = coor_to_voxelidx.mutable_data<int>();
  std::fill(coor_to_voxelidx_data,
            coor_to_voxelidx_data + coor_to_voxelidx.size(),
            static_cast<int>(-1));

  // coors, num_points_per_voxel, coor_to_voxelidx are int Tensor
  // printf("cpu coor_to_voxelidx size: [%d, %d, %d]\n", grid_size[2],
  // grid_size[1], grid_size[0]);

  auto voxel_num = paddle::Tensor(paddle::PlaceType::kCPU, {1});
  auto *voxel_num_data = voxel_num.mutable_data<int>();
  voxel_num_data[0] = 0;
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "hard_voxelize_cpu_kernel", ([&] {
        hard_voxelize_cpu_kernel<data_t, int>(
            points.data<data_t>(), voxels.mutable_data<data_t>(), coors_data,
            num_points_per_voxel_data, coor_to_voxelidx_data, voxel_num_data,
            voxel_size, coors_range, grid_size, max_points, max_voxels,
            num_points, num_features, NDim, voxels.shape(), coors.shape(),
            num_points_per_voxel.shape(), coor_to_voxelidx.shape());
      }));
  return {voxel_num, voxels, coors, num_points_per_voxel};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> dynamic_voxelize_cuda(
    const paddle::Tensor &points, const std::vector<float> &voxel_size,
    const std::vector<float> &coors_range, const int NDim = 3);

std::vector<paddle::Tensor> hard_voxelize_cuda(
    const paddle::Tensor &points, const std::vector<float> &voxel_size,
    const std::vector<float> &coors_range, int max_points, int max_voxels,
    int NDim = 3);
#endif

std::vector<paddle::Tensor> dynamic_voxelize(
    const paddle::Tensor &points, const std::vector<float> &voxel_size,
    const std::vector<float> &coors_range, const int NDim = 3) {
  if (points.place() == paddle::PlaceType::kCPU) {
    return dynamic_voxelize_cpu(points, voxel_size, coors_range, NDim);
#ifdef PADDLE_WITH_CUDA
  } else if (points.place() == paddle::PlaceType::kGPU) {
    return dynamic_voxelize_cuda(points, voxel_size, coors_range, NDim);
#endif
  } else {
    PD_THROW(
        "Unsupported device type for dynamic_voxelize "
        "operator.");
  }
}

std::vector<paddle::Tensor> hard_voxelize(const paddle::Tensor &points,
                                          const std::vector<float> &voxel_size,
                                          const std::vector<float> &coors_range,
                                          int max_points, int max_voxels,
                                          int NDim = 3) {
  if (points.place() == paddle::PlaceType::kCPU) {
    return hard_voxelize_cpu(points, voxel_size, coors_range, max_points,
                             max_voxels, NDim);
#ifdef PADDLE_WITH_CUDA
  } else if (points.place() == paddle::PlaceType::kGPU) {
    return hard_voxelize_cuda(points, voxel_size, coors_range, max_points,
                              max_voxels, NDim);
#endif
  } else {
    PD_THROW(
        "Unsupported device type for hard_voxelize "
        "operator.");
  }
}

std::vector<std::vector<int64_t>> DynamicInferShape(
    std::vector<int64_t> points_shape) {
  return {{points_shape[0], 3}};
}

std::vector<paddle::DataType> DynamicInferDtype(paddle::DataType points_dtype) {
  return {points_dtype};
}

std::vector<std::vector<int64_t>> HardInferShape(
    std::vector<int64_t> points_shape, const std::vector<float> &voxel_size,
    const std::vector<float> &coors_range, const int &max_points,
    const int &max_voxels, const int &NDim = 3) {
  return {{1},
          {max_voxels, max_points, points_shape[1]},
          {max_voxels, 3},
          {max_voxels}};
}

std::vector<paddle::DataType> HardInferDtype(paddle::DataType points_dtype) {
  return {paddle::DataType::INT32, points_dtype, paddle::DataType::INT32,
          paddle::DataType::INT32};
}

PD_BUILD_OP(dynamic_voxelize)
    .Inputs({"POINTS"})
    .Outputs({"COORS"})
    .SetKernelFn(PD_KERNEL(dynamic_voxelize))
    .Attrs({"voxel_size: std::vector<float>", "coors_range: std::vector<float>",
            "NDim: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(DynamicInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DynamicInferDtype));

PD_BUILD_OP(hard_voxelize)
    .Inputs({"POINTS"})
    .Outputs({"VOXEL_NUM", "VOXELS", "COORS", "NUM_POINTS_PER_VOXEL"})
    .SetKernelFn(PD_KERNEL(hard_voxelize))
    .Attrs({"voxel_size: std::vector<float>", "coors_range: std::vector<float>",
            "max_points: int", "max_voxels: int", "NDim: int"})
    .SetInferShapeFn(PD_INFER_SHAPE(HardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(HardInferDtype));
