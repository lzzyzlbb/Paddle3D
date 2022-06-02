/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <numeric>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <time.h>

#include "paddle/include/paddle_inference_api.h"

DEFINE_bool(use_trt, true, "use trt.");
DEFINE_string(trt_precision, "trt_fp32", "trt_fp32, trt_fp16, etc.");
DEFINE_bool(tune, false, "tune to get shape range.");

using paddle_infer::Config;
using paddle_infer::Predictor;
using paddle_infer::CreatePredictor;

const std::string shape_range_info = "shape_range_info.pbtxt";

paddle_infer::PrecisionType GetPrecisionType(const std::string& ptype) {
  if (ptype == "trt_fp32")
    return paddle_infer::PrecisionType::kFloat32;
  if (ptype == "trt_fp16")
    return paddle_infer::PrecisionType::kHalf;
  return paddle_infer::PrecisionType::kFloat32;
}

void run(Predictor *predictor, 
         const std::vector<int> &images_shape,
         const std::vector<float> &images_data,
         const std::vector<int> &cam_shape,
         const std::vector<float> &cam_data,
         const std::vector<int> &lidar_shape,
         const std::vector<float> &lidar_data,
         std::vector<float> *boxes,
         std::vector<float> *labels,
         std::vector<float> *scores) {
  
  auto input_names = predictor->GetInputNames();
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor->GetInputHandle(tensor_name);
    if (tensor_name == "images") {
      in_tensor->Reshape(images_shape);
      in_tensor->CopyFromCpu(images_data.data());
    } else if (tensor_name == "trans_cam_to_img") {
      in_tensor->Reshape(cam_shape);
      in_tensor->CopyFromCpu(cam_data.data());
    } else if (tensor_name == "trans_lidar_to_cam") {
      in_tensor->Reshape(lidar_shape);
      in_tensor->CopyFromCpu(lidar_data.data());
    }
  }

  clock_t start,stop;
  start = clock();
  for (int i = 0; i < 100; i++) {
    CHECK(predictor->Run());

    auto output_names = predictor->GetOutputNames();
    for (size_t i = 0; i != output_names.size(); i++) {
      auto output = predictor->GetOutputHandle(output_names[i]);
      std::vector<int> output_shape = output->shape();
      int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                    std::multiplies<int>());
      if (i == 0) {
        boxes->resize(out_num);
        output->CopyToCpu(boxes->data());
      } else if (i == 1) {
        labels->resize(out_num);
        output->CopyToCpu(labels->data());
      } else if (i == 2) {
        scores->resize(out_num);
        output->CopyToCpu(scores->data());
      }
    }
  }
  stop = clock();
  double endtime = (double)(stop - start) / CLOCKS_PER_SEC;
  LOG(INFO) << "time:" << endtime << "s";
}

int main() {
  paddle::AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel("caddn_infer_model/model.pdmodel",
                  "caddn_infer_model/model.pdiparams");
  // use trt
  if (FLAGS_use_trt) {
    config.EnableTensorRtEngine(1 << 30, 8, 3,
                              GetPrecisionType(FLAGS_trt_precision), false, false);
    config.EnableTunedTensorRtDynamicShape(shape_range_info, true);
    if (FLAGS_tune) {
      config.CollectShapeRangeInfo(shape_range_info);
    }
  }

  auto predictor{paddle_infer::CreatePredictor(config)};
  std::vector<int> images_shape = {1, 3, 375, 1242};
  std::vector<float> images_data(1 * 3 * 375 * 1242, 1);
  std::vector<int> cam_shape = {1, 3, 4};
  std::vector<float> cam_data(1 * 3 * 4, 1);
  std::vector<int> lidar_shape = {1, 4, 4};
  std::vector<float> lidar_data(1 * 4 * 4, 1);
  std::vector<float> boxes;
  std::vector<float> labels;
  std::vector<float> scores;
  run(predictor.get(), images_shape, images_data, 
                       cam_shape, cam_data,
                       lidar_shape, lidar_data,
                       &boxes, &labels, &scores);
  for (auto e : boxes) {
    LOG(INFO) << e;
  }
  for (auto e : labels) {
    LOG(INFO) << e;
  }
  for (auto e : scores) {
    LOG(INFO) << e;
  }
  return 0;
}
