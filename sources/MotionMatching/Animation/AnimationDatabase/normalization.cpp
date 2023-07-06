#include "normalization.h"

float sqrf(float f)
{
  return f * f;
}

//based on idea from github.com/orangeduck/Motion-Matching

void normalize_feature(
    std::vector<std::array<float, FrameFeature::frameSize>> features,
    std::array<float, FrameFeature::frameSize> featuresScale,
    std::array<float, FrameFeature::frameSize> featuresMean,
    int offset, 
    int size, 
    float weight)
{
  for (int j = 0; j < size; j++)
  {
    featuresMean[offset + j] = 0.f;    
  }
  for (unsigned int i = 0; i < features.size(); i++)
  {
    for (int j = 0; j < size; j++)
    {
      featuresMean[offset + j] += features[i][offset + j] / features.size();
    }
  }

  std::vector<float> variations(size, 0.0);

  for (unsigned int i = 0; i < features.size(); i++)
  {
    for (int j = 0; j < size; j++)
    {
      variations[j] += sqrf(features[i][offset + j] - featuresMean[offset + j]) / features.size();
    }
  }
  float std = 0.f;
  for (int j = 0; j < size; j++)
  {
    std += sqrf(variations[j]) / size;
  }
  for (int j = 0; j < size; j++)
  {
    featuresScale[offset + j] = std / weight;
  }
  for (unsigned int i = 0; i < features.size(); i++)
  {
    for (int j = 0; j < size; j++)
    {
      features[i][offset + j] = (features[i][offset + j] - featuresMean[offset + j]) / featuresScale[offset + j];
    }
  }
}