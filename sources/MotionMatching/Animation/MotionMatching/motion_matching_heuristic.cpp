#include "motion_matching.h"
#include "../settings.h"

struct ArgMin
{
  float value;
  uint clip, frame;
  MatchingScores score;
};

static float sqrf(float f)
{
  return f * f;
}

float clampf(float value, float min_bound, float max_bound)
{
  if (value > max_bound)
  {
    return max_bound;
  }
  if (value < min_bound)
  {
    return min_bound;
  }
  return value;
}



void normalize_frame(
  std::array<float, FrameFeature::frameSize>& query, 
  std::array<float, FrameFeature::frameSize>& n_query,
  boundingStructure& bounding)
{
  for (int i = 0; i < FrameFeature::frameSize; ++i)
  {
    n_query[i] = (query[i] - bounding.featuresMean[i]) / bounding.featuresScale[i];
  }
}

AnimationIndex solve_motion_matching_heuristic(
  AnimationDataBasePtr dataBase,
  const AnimationIndex &index,
  const AnimationGoal &goal,
  MatchingScores &best_score,
  const MotionMatchingSettings &mmsettings)
{
  if (!dataBase || !index())
    return AnimationIndex();
  uint curClip = index.get_clip_index();
  uint curCadr = index.get_cadr_index();

  boundingStructure& bounding = dataBase->bounding;
  ArgMin best = {INFINITY, curClip, curCadr, best_score};
  std::vector<AnimationClip>& clips = dataBase->clips;

  float score = 0;
  std::array<float, FrameFeature::frameSize> goalFeature;
  goal.feature.save_to_array(goalFeature);
  for (int i = 0; i < FrameFeature::frameSize; ++i)
  {
    goalFeature[i] = (goalFeature[i] - bounding.featuresMean[i]) / bounding.featuresScale[i];
  }

  for (uint nextClip = 0; nextClip < clips.size(); nextClip++)
  {
    const AnimationClip &clip = clips[nextClip];
    if (!has_goal_tags(goal.tags, clip.tags))
      continue;
    int frameIdx = dataBase->clipsStarts[nextClip];
    while (frameIdx < dataBase->clipsStarts[nextClip + 1])
    {
      int lrBoxIdx = frameIdx / boundingStructure::LR_BOX_SIZE;
      int nextBoxFrame = (lrBoxIdx + 1) * boundingStructure::LR_BOX_SIZE;
      for (int i = 0; i < FrameFeature::frameSize; ++i)
      {
        score += sqrf(goalFeature[i] - clampf(goalFeature[i], bounding.lrBoxMin[lrBoxIdx][i], bounding.lrBoxMax[lrBoxIdx][i]));
      }
    }
  }

  best_score = best.score;
  return AnimationIndex(dataBase, best.clip, best.frame);
}