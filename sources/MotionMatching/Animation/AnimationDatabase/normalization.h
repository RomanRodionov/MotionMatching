#pragma once
#include <common.h>
#include <vector>
#include "animation_feature.h"
#include "animation_database.h"

void normalize_feature(
    std::vector<std::array<float, FrameFeature::frameSize>> features,
    std::array<float, FrameFeature::frameSize> featuresScale,
    std::array<float, FrameFeature::frameSize> featuresMean,
    const int offset, 
    const int size, 
    const float weight = 1.0f);