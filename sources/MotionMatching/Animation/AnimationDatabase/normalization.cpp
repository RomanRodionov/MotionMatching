#include "normalization.h"

static float sqrf(float f)
{
  return f * f;
}

inline float minf(float a, float b)
{
  return (a > b) ? b : a;
}

inline float maxf(float a, float b)
{
  return (a > b) ? a : b;
}

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

void boundingStructure::find_boxes_values(std::vector<std::array<float, FrameFeature::frameSize>> features)
{
  int smBoxNum = ((features.size() + SM_BOX_SIZE - 1) / SM_BOX_SIZE);
  int lrBoxNum = ((features.size() + LR_BOX_SIZE - 1) / LR_BOX_SIZE);

  std::array<float, FrameFeature::frameSize> posInf = {+FLT_MAX};
  std::array<float, FrameFeature::frameSize> negInf = {+FLT_MAX};
  smBoxMin = std::vector<std::array<float, FrameFeature::frameSize>>(smBoxNum, posInf);
  smBoxMax = std::vector<std::array<float, FrameFeature::frameSize>>(smBoxNum, negInf);
  lrBoxMin = std::vector<std::array<float, FrameFeature::frameSize>>(lrBoxNum, posInf);
  lrBoxMax = std::vector<std::array<float, FrameFeature::frameSize>>(lrBoxNum, negInf);

  for (unsigned int i = 0; i < features.size(); i++)
  {
    int smIdx = i / SM_BOX_SIZE;
    int lrIdx = i / LR_BOX_SIZE;
    for (int j = 0; j < FrameFeature::frameSize; j++)
    {
      smBoxMin[smIdx][j] = minf(smBoxMin[smIdx][j], features[i][j]);
      smBoxMax[smIdx][j] = maxf(smBoxMax[smIdx][j], features[i][j]);
      lrBoxMin[lrIdx][j] = minf(lrBoxMin[lrIdx][j], features[i][j]);
      lrBoxMax[lrIdx][j] = maxf(lrBoxMax[lrIdx][j], features[i][j]);
    }
  }
}