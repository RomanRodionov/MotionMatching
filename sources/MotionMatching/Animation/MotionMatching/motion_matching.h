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

void store_database(AnimationDataBasePtr dataBase, const MotionMatchingSettings &mmsettings, vector<uint> &labels, uint feature_ssbo, int &size);
void store_goal_feature(const AnimationGoal& goal, const MotionMatchingSettings &mmsettings, uint uboBlock);