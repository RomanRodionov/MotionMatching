#include "animation_feature.h"
#include "animation_nodes.h"
#include "../settings.h"
#include <cmath>
#include <cstdlib>
#include <map>

#define FEATURE(node) {#node, (int)AnimationFeaturesNode::node}, {#node "Speed",(int)AnimationFeaturesNode::Count + (int)AnimationFeaturesNode::node}
    
std::map<std::string, int> featureMap = {
  FEATURE(LeftHand),
  FEATURE(RightHand),
  FEATURE(LeftToeBase),
  FEATURE(RightToeBase)
};
#undef FEATURE



void NodeFeatures::set_feature(const string& name, vec3 feature)
{
  auto it = featureMap.find(name);
  if (it != featureMap.end())
    nodes[(int)it->second] = feature;
  
}
float pose_matching_norma(const NodeFeatures& feature1, const NodeFeatures& feature2, const MotionMatchingSettings &settings)
{
  float pose_norma = 0.f, vel_norma = 0.f;
  if (settings.applySettingsOnce)
  {
    for (int i = 0; i < (int)AnimationFeaturesNode::Count; i++)
    {
      pose_norma += length(feature1.nodes[i] - feature2.nodes[i]);
      vel_norma += length(feature1.nodesVelocity[i] - feature2.nodesVelocity[i]);
    }
    return pose_norma + vel_norma;
  }
  else
  {
    for (int i = 0; i < (int)AnimationFeaturesNode::Count; i++)
    {
      pose_norma += settings.nodeWeights[i] * length(feature1.nodes[i] - feature2.nodes[i]);
      vel_norma += settings.velocityMatching ? settings.velocitiesWeights[i] * length(feature1.nodesVelocity[i] - feature2.nodesVelocity[i]) : 0.f;
    }
  }
  return settings.poseMatchingWeight * pose_norma + settings.velocityMatchingWeight * vel_norma;
}
bool has_goal_tags(AnimationTags goal, AnimationTags clips_tag)
{
  return goal.tags == clips_tag.tags;
}

float goal_tag_norma(AnimationTags /* goal */, AnimationTags /* clips_tag */)
{
  return 0;
}

float goal_path_norma(const AnimationTrajectory &path, const AnimationTrajectory &goal)
{
  float path_norma = 0.f;
  float distScale = length(goal.trajectory.back().point + path.trajectory.back().point) * 0.5f;
  for (uint i = 0; i < AnimationTrajectory::PathLength; i++)
    path_norma += length((path.trajectory[i].point - goal.trajectory[i].point));
  return  path_norma / (0.1f + distScale);
}
float trajectory_v_norma(const AnimationTrajectory &path, const AnimationTrajectory &goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < AnimationTrajectory::PathLength; i++)
    path_norma += length((path.trajectory[i].velocity - goal.trajectory[i].velocity));
  return  path_norma;
}
float trajectory_w_norma(const AnimationTrajectory &path, const AnimationTrajectory &goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < AnimationTrajectory::PathLength; i++)
    path_norma += length((path.trajectory[i].angularVelocity - goal.trajectory[i].angularVelocity));
  return  path_norma;
}
MatchingScores get_score(const FrameFeature& clip_feature, const FrameFeature& goal_feature,
  const MotionMatchingSettings &settings)
{
  MatchingScores score{0, 0, 0, 0, 0, 0};
  score.pose = pose_matching_norma(clip_feature.features, goal_feature.features, settings);
  score.goal_path = goal_path_norma(clip_feature.trajectory, goal_feature.trajectory) * settings.goalPathMatchingWeight;
  if (settings.applySettingsOnce)
  {
    score.trajectory_v = trajectory_v_norma(clip_feature.trajectory, goal_feature.trajectory);
    score.trajectory_w = trajectory_w_norma(clip_feature.trajectory, goal_feature.trajectory);
  }
  else
  {
    score.trajectory_v = trajectory_v_norma(clip_feature.trajectory, goal_feature.trajectory) * settings.goalVelocityWeight;
    score.trajectory_w = trajectory_w_norma(clip_feature.trajectory, goal_feature.trajectory) * settings.goalAngularVelocityWeight;
  }
  score.full_score = score.pose * settings.realism + (score.goal_path + score.trajectory_v + score.trajectory_w);

  return score;
}

float path_norma(const AnimationTrajectory& clip_feature, const AnimationTrajectory &goal_feature, const MotionMatchingSettings &settings)
{
  if (settings.applySettingsOnce)
  {
      return 
      goal_path_norma(clip_feature, goal_feature)
        +
      trajectory_v_norma(clip_feature, goal_feature)
        +
      trajectory_w_norma(clip_feature, goal_feature);
  }
  return 
  goal_path_norma(clip_feature, goal_feature) * settings.goalPathMatchingWeight
    +
  trajectory_v_norma(clip_feature, goal_feature) * settings.goalVelocityWeight
    +
  trajectory_w_norma(clip_feature, goal_feature) * settings.goalAngularVelocityWeight;
}

float FrameMetric::distance(const FrameFeature& clip_feature, const FrameFeature &goal_feature) const
{
  if (settings.applySettingsOnce)
  {
      pose_matching_norma(clip_feature.features, goal_feature.features, settings)
        +
      goal_path_norma(clip_feature.trajectory, goal_feature.trajectory)
        +
      trajectory_v_norma(clip_feature.trajectory, goal_feature.trajectory)
        +
      trajectory_w_norma(clip_feature.trajectory, goal_feature.trajectory);
  }
  return 
  pose_matching_norma(clip_feature.features, goal_feature.features, settings) * settings.realism
    +
  goal_path_norma(clip_feature.trajectory, goal_feature.trajectory) * settings.goalPathMatchingWeight
    +
  trajectory_v_norma(clip_feature.trajectory, goal_feature.trajectory) * settings.goalVelocityWeight
    +
  trajectory_w_norma(clip_feature.trajectory, goal_feature.trajectory) * settings.goalAngularVelocityWeight;
}

void FrameFeature::get_sizes(std::vector<int>& featuresSizes)
{
  for (int i = 0; i < (int)AnimationFeaturesNode::Count * 2; ++i)
  {
    featuresSizes.push_back(3);
  }
  
  for (int i = 0; i < (int)AnimationTrajectory::PathLength; ++i)
  {
    featuresSizes.push_back(3);
    featuresSizes.push_back(3);
    featuresSizes.push_back(1);
  }
}

void FrameFeature::save_to_array(std::array<float, FrameFeature::frameSize>& featuresArray) const
{
  int offset = 0;
  for (int i = 0; i < (int)AnimationFeaturesNode::Count; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      featuresArray[offset++] = features.nodes[i][j];
    }
    for (int j = 0; j < 3; ++j)
    {
      featuresArray[offset++] = features.nodesVelocity[i][j];
    }
  }
  for (int i = 0; i < (int)AnimationTrajectory::PathLength; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      featuresArray[offset++] = trajectory.trajectory[i].point[j];
    }
    for (int j = 0; j < 3; ++j)
    {
      featuresArray[offset++] = trajectory.trajectory[i].velocity[j];
    }
    featuresArray[offset++] = trajectory.trajectory[i].angularVelocity;
  }
}