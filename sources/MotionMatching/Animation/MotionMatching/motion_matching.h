#pragma once
#include "common.h"
#include "../animation_index.h"
#include <vector>

enum class MotionMatchingSolverType
{
  BruteForce,
  VPTree,
  CoverTree,
  KDTree,
  CSBruteForce,
  HeuristicBruteForce,
  Count
};

struct Settings;
struct MotionMatchingSettings;
struct MotionMatchingOptimisationSettings;

class MotionMatching
{
public:
  AnimationDataBasePtr dataBase;
  AnimationLerpedIndex index;
  float skip_time;
  int lod;
  MatchingScores bestScore;
  MotionMatching(AnimationDataBasePtr dataBase, AnimationLerpedIndex index);
  MotionMatching() = default;
  AnimationLerpedIndex get_index() const;

  AnimationDataBasePtr get_data_base() const;
};

struct FeatureCell
{
  vec4 nodes[(uint)AnimationFeaturesNode::Count];
  vec4 nodesVelocity[(uint)AnimationFeaturesNode::Count];
  vec4 points[(uint)AnimationTrajectory::PathLength];
  vec4 pointsVelocity[(uint)AnimationTrajectory::PathLength];
  vec4 angularVelocity;
  Tag tags;
  uint padding1;
  uint padding2;
};

struct ShaderMatchingScores
{
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint idx;
  uint padding;
};

void store_database(AnimationDataBasePtr dataBase, const MotionMatchingSettings &mmsettings, vector<uint> &labels, uint feature_ssbo, int &size);
void pack_goal_feature(const AnimationGoal& goal, const MotionMatchingSettings &mmsettings, FeatureCell& goal_feature);