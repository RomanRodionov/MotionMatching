#include "motion_matching.h"
#include "../settings.h"

struct ArgMin
{
  float value;
  uint clip, frame;
  MatchingScores score;
};

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
  
  ArgMin best = {INFINITY, curClip, curCadr, best_score};

  std::vector<AnimationClip>& clips = dataBase->clips;

  best_score = best.score;
  return AnimationIndex(dataBase, best.clip, best.frame);
}