#include "voxelize_op.h"
#include "paddle/extension.h"

#define CHECK_INPUT_CUDA(x) \
  PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

HOST_DEVICE_INLINE int CeilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void fill(const int nthreads, T* empty_tensor, const int& value) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    empty_tensor[thread_idx] = static_cast<T>(value);
  }
}

template <typename T>
void fill(T* empty_tensor, const size_t& size, const int& value) {
  cudaMemset(empty_tensor, static_cast<T>(value), size * sizeof(T));
}

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int index = thread_idx / num_features;

    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

template <typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    // if (index >= num_points) return;
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if ((index >= num_points) || (coor_offset[0] == -1)) return;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          // point to the same coor that first show up
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          // out of boundary
          return;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
  }
}

template <typename T_int>
__global__ void determin_voxel_num(
    // const T_int* coor,
    T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,
    const int max_points, const int max_voxels, const int num_points) {
  // only calculate the coors before this coor[index]
  for (int i = 0; i < num_points; ++i) {
    // if (coor[i][0] == -1)
    //    continue;
    int point_pos_in_voxel = point_to_voxelidx[i];
    // record voxel
    if (point_pos_in_voxel == -1) {
      // out of max_points or invalid point
      continue;
    } else if (point_pos_in_voxel == 0) {
      // record new voxel
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

std::vector<paddle::Tensor> dynamic_voxelize_cuda(
    const paddle::Tensor& points, const std::vector<float>& voxel_size,
    const std::vector<float>& coors_range, const int NDim = 3) {
  // check device
  CHECK_INPUT_CUDA(points);

  auto num_points = points.shape()[0];
  auto num_features = points.shape()[1];

  auto coors = paddle::Tensor(paddle::PlaceType::kGPU, {num_points, 3});
  auto* coors_data = coors.mutable_data<int>();
  fill<int>(coors_data, coors.size(), 0);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  const int col_blocks = CeilDiv(num_points, threadsPerBlock);
  dim3 blocks(col_blocks);
  dim3 threads(threadsPerBlock);

  PD_DISPATCH_FLOATING_TYPES(points.type(), "dynamic_voxelize_kernel", [&] {
    dynamic_voxelize_kernel<data_t, int>
        <<<blocks, threads, 0, points.stream()>>>(
            points.data<data_t>(), coors_data, voxel_x, voxel_y, voxel_z,
            coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
            coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
            NDim);
  });
  cudaDeviceSynchronize();

  return {coors};
}

std::vector<paddle::Tensor> hard_voxelize_cuda(
    const paddle::Tensor& points, const std::vector<float>& voxel_size,
    const std::vector<float>& coors_range, int max_points, int max_voxels,
    int NDim = 3) {
  // check device
  CHECK_INPUT_CUDA(points);

  auto num_points = points.shape()[0];
  auto num_features = points.shape()[1];

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  auto temp_coors = paddle::Tensor(paddle::PlaceType::kGPU, {num_points, NDim});
  auto* temp_coors_data = temp_coors.mutable_data<int>();
  fill<int>(temp_coors_data, temp_coors.size(), 0);

  dim3 grid(std::min(CeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  PD_DISPATCH_FLOATING_TYPES(points.type(), "dynamic_voxelize_kernel", ([&] {
                               dynamic_voxelize_kernel<data_t, int>
                                   <<<grid, block, 0, points.stream()>>>(
                                       points.data<data_t>(), temp_coors_data,
                                       voxel_x, voxel_y, voxel_z, coors_x_min,
                                       coors_y_min, coors_z_min, coors_x_max,
                                       coors_y_max, coors_z_max, grid_x, grid_y,
                                       grid_z, num_points, num_features, NDim);
                             }));
  cudaDeviceSynchronize();

  // 2. map point to the idx of the corresponding voxel, find duplicate coor
  // create some temporary variables
  auto point_to_pointidx =
      paddle::Tensor(paddle::PlaceType::kGPU, {num_points});
  auto* point_to_pointidx_data = point_to_pointidx.mutable_data<int>();
  fill<int>(point_to_pointidx_data, point_to_pointidx.size(), -1);

  auto point_to_voxelidx =
      paddle::Tensor(paddle::PlaceType::kGPU, {num_points});
  auto* point_to_voxelidx_data = point_to_voxelidx.mutable_data<int>();
  fill<int>(point_to_voxelidx_data, point_to_voxelidx.size(), -1);

  dim3 map_grid(std::min(CeilDiv(num_points, 512), 4096));
  dim3 map_block(512);
  point_to_voxelidx_kernel<int><<<map_grid, map_block, 0>>>(
      temp_coors_data, point_to_voxelidx_data, point_to_pointidx_data,
      max_points, max_voxels, num_points, NDim);
  cudaDeviceSynchronize();

  // 3. determin voxel num and voxel's coor index
  // make the logic in the CUDA device could accelerate about 10 times
  auto coor_to_voxelidx = paddle::Tensor(paddle::PlaceType::kGPU, {num_points});
  auto* coor_to_voxelidx_data = coor_to_voxelidx.mutable_data<int>();
  fill<int>(coor_to_voxelidx_data, coor_to_voxelidx.size(), -1);

  auto voxel_num = paddle::Tensor(paddle::PlaceType::kGPU, {1});
  auto* voxel_num_data = voxel_num.mutable_data<int>();
  fill<int>(voxel_num_data, voxel_num.size(), 0);

  auto num_points_per_voxel =
      paddle::Tensor(paddle::PlaceType::kGPU, {max_voxels});
  auto* num_points_per_voxel_data = num_points_per_voxel.mutable_data<int>();
  fill<int>(num_points_per_voxel_data, num_points_per_voxel.size(), 0);

  determin_voxel_num<int>
      <<<1, 1, 0>>>(num_points_per_voxel_data, point_to_voxelidx_data,
                    point_to_pointidx_data, coor_to_voxelidx_data,
                    voxel_num_data, max_points, max_voxels, num_points);
  cudaDeviceSynchronize();

  // 4. copy point features to voxels
  // Step 4 & 5 could be parallel
  auto voxels = paddle::Tensor(paddle::PlaceType::kGPU,
                               {max_voxels, max_points, num_features});
  dim3 fill_grid(std::min(CeilDiv(voxels.size(), 512), 4096));
  dim3 fill_block(512);
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "fill",
      ([&] { fill<data_t>(voxels.mutable_data<data_t>(), voxels.size(), 0); }));

  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(CeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  PD_DISPATCH_FLOATING_TYPES(
      points.type(), "assign_point_to_voxel", ([&] {
        assign_point_to_voxel<data_t, int>
            <<<cp_grid, cp_block, 0, points.stream()>>>(
                pts_output_size, points.data<data_t>(), point_to_voxelidx_data,
                coor_to_voxelidx_data, voxels.mutable_data<data_t>(),
                max_voxels, max_points, num_features, num_points, NDim);
      }));

  // 5. copy coors of each voxels
  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(std::min(CeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);

  auto coors = paddle::Tensor(paddle::PlaceType::kGPU, {max_voxels, 3});
  auto* coors_data = coors.mutable_data<int>();
  fill<int>(coors_data, coors.size(), 0);

  assign_voxel_coors<int><<<coors_cp_grid, coors_cp_block, 0>>>(
      coors_output_size, temp_coors_data, point_to_voxelidx_data,
      coor_to_voxelidx_data, coors_data, num_points, NDim);
  cudaDeviceSynchronize();

  return {voxel_num, voxels, coors, num_points_per_voxel};
}
