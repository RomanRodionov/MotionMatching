#pragma once
#include <common.h>
#include <vector>
#include "animation_feature.h"

//based on idea from github.com/orangeduck/Motion-Matching

void normalize_feature(
    std::vector<std::array<float, FrameFeature::frameSize>> features,
    std::array<float, FrameFeature::frameSize> featuresScale,
    std::array<float, FrameFeature::frameSize> featuresMean,
    const int offset, 
    const int size, 
    const float weight = 1.0f);

struct boundingStructure
{
  std::array<float, FrameFeature::frameSize> featuresScale;
  std::array<float, FrameFeature::frameSize> featuresMean;

  std::vector<std::array<float, FrameFeature::frameSize>> smBoxMin, smBoxMax, lrBoxMin, lrBoxMax;
  static const int LR_BOX_SIZE = 64, SM_BOX_SIZE = 16;
  void find_boxes_values(std::vector<std::array<float, FrameFeature::frameSize>> features);
};