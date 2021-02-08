#include "animation_feature_weight.h"
#define FEATURE(F) {#F, AnimationFeaturesNode::F}
AnimationFeaturesWeights::AnimationFeaturesWeights():
norma_function_weight(1.f), weights((int)AnimationFeaturesNode::Count, 1.f), featureMap({
    FEATURE(Hips),
    FEATURE(LeftForeArm),
    FEATURE(LeftHand),
    FEATURE(RightForeArm),
    FEATURE(RightHand),
    FEATURE(LeftLeg),
    FEATURE(LeftFoot),
    FEATURE(RightLeg),
    FEATURE(RightFoot),
    FEATURE(Head)})
{}
size_t AnimationFeaturesWeights::serialize(std::ostream& os) const
{
  size_t size = 0;
  size += write(os, featureMap.size());
  for (const auto &p : featureMap)
  {
    size += write(os, p.first);
    size += write(os, weights[(int)p.second]);
  }
  size += write(os, norma_function_weight);
  return size;
}
size_t AnimationFeaturesWeights::deserialize(std::istream& is)
{
  size_t size = 0;
  size_t n = 0;
  size += read(is, n);
  for (uint i = 0; i < n; i++)
  {
    string name; 
    float weight;
    size += read(is, name);
    size += read(is, weight);
    weights[(int)featureMap[name]] = weight;
  }
  size += read(is, norma_function_weight);
  return size;
}