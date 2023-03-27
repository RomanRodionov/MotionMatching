#include "motion_matching.h"
#include "../settings.h"
#include <render/shader/shader.h>
#include <profiler/profiler.h>

void store_database(AnimationDataBasePtr dataBase, const MotionMatchingSettings &mmsettings, vector<uint> &labels, uint feature_ssbo, int &size)
{
  float poseWeight = mmsettings.poseMatchingWeight;
  float velocityWeight = mmsettings.velocityMatchingWeight;
  std::vector<FeatureCell> featureData;
  uint featuresCounter = 0;
  size = 0;
  for (uint nextClip = 0; nextClip < dataBase->clips.size(); nextClip++)
  {
    const AnimationClip &clip = dataBase->clips[nextClip];
    labels.push_back(featuresCounter);
    featuresCounter += clip.duration;
    for (uint nextCadr = 0, n = clip.duration; nextCadr < n; nextCadr++)
    {
      auto &frame = clip.features[nextCadr];
      FeatureCell nextFeatureCell;
      for (uint node = 0; node < (uint)AnimationFeaturesNode::Count; node++)
      {
        nextFeatureCell.nodes[node] = vec4(frame.features.nodes[node] * float(mmsettings.nodeWeights[node]) * poseWeight, 0);
        if (mmsettings.velocityMatching)
          nextFeatureCell.nodesVelocity[node] = vec4(frame.features.nodesVelocity[node] * float(mmsettings.velocitiesWeights[node]) * velocityWeight, 0);
        else
          nextFeatureCell.nodesVelocity[node] = vec4(0, 0, 0, 0);
      }
      for (uint point = 0; point < (uint)AnimationTrajectory::PathLength; point++)
      {
        nextFeatureCell.points[point] = vec4(frame.trajectory.trajectory[point].point, 0);
        nextFeatureCell.pointsVelocity[point] = vec4(frame.trajectory.trajectory[point].velocity * mmsettings.goalVelocityWeight, 0);
        nextFeatureCell.angularVelocity[point] = frame.trajectory.trajectory[point].angularVelocity * mmsettings.goalAngularVelocityWeight;
      }
      nextFeatureCell.tags = clip.tags.tags;
      featureData.push_back(nextFeatureCell);
      size++;
    }
  }
  store_ssbo(feature_ssbo, featureData.data(), sizeof(FeatureCell) * featuresCounter, GL_STATIC_DRAW);
}

void pack_goal_feature(const AnimationGoal& goal, const MotionMatchingSettings &mmsettings, FeatureCell& goal_feature)
{
  float poseWeight = mmsettings.poseMatchingWeight;
  float velocityWeight = mmsettings.velocityMatchingWeight;
  for (uint node = 0; node < (uint)AnimationFeaturesNode::Count; node++)
  {
    goal_feature.nodes[node] = vec4(goal.feature.features.nodes[node] * float(mmsettings.nodeWeights[node]) * poseWeight, 0);
    if (mmsettings.velocityMatching)
      goal_feature.nodesVelocity[node] = vec4(goal.feature.features.nodesVelocity[node] * float(mmsettings.velocitiesWeights[node]) * velocityWeight, 0);
    else
      goal_feature.nodesVelocity[node] = vec4(0, 0, 0, 0);
  }
  for (uint point = 0; point < (uint)AnimationTrajectory::PathLength; point++)
  {
    goal_feature.points[point] = vec4(goal.feature.trajectory.trajectory[point].point, 0);
    goal_feature.pointsVelocity[point] = vec4(goal.feature.trajectory.trajectory[point].velocity * mmsettings.goalVelocityWeight, 0);
    goal_feature.angularVelocity[point] = goal.feature.trajectory.trajectory[point].angularVelocity * mmsettings.goalAngularVelocityWeight;
  }
  goal_feature.tags = goal.tags.tags;
}