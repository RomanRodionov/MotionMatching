#include "motion_matching.h"
#include "../settings.h"
#include <render/shader/shader.h>

struct ArgMin
{
  float value;
  uint clip, frame;
  MatchingScores score;
};

static ArgMin mm_min2(const ArgMin &a, const ArgMin &b)
{
  return a.value < b.value ? a : b; 
}

AnimationIndex solve_motion_matching_cs(
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
  //#pragma omp declare reduction(mm_min: ArgMin: omp_out=mm_min2(omp_out, omp_in))\
  //  initializer(omp_priv={INFINITY, 0, 0,{0,0,0,0,0,0}})
  //#pragma omp parallel for reduction(mm_min:best)
  for (uint nextClip = 0; nextClip < dataBase->clips.size(); nextClip++)
  {
    const AnimationClip &clip = dataBase->clips[nextClip];

    if (!has_goal_tags(goal.tags, clip.tags))
      continue;
    for (uint nextCadr = 0, n = clip.duration; nextCadr < n; nextCadr++)
    {
      MatchingScores score = get_score(clip.features[nextCadr], goal.feature, mmsettings);
      
      float matching = score.full_score;
      ArgMin cur = {matching, nextClip, nextCadr, score};
      best = mm_min2(best, cur);
    }
  }
  //
  // parallel reduction for finding the minimum value in an array of positive floats
  int group_size = 512;
  uint arr_size = 256;
  static uint a = 0;
  GLfloat *data = new GLfloat[arr_size];
	for (uint i = 0; i < arr_size; ++i) {
		data[i] = i + 100;
  }
  data[a % 256] = 15;
  a++;
  auto compute_shader = get_compute_shader("compute_motion");
  compute_shader.use();
  uint ssbo = compute_shader.create_ssbo();
  compute_shader.store_ssbo(ssbo, data, arr_size * sizeof(float));
  delete[] data;
  compute_shader.set_int("arr_size", arr_size);
  uint dsize = arr_size;
  if (dsize % 2 > 0) dsize++;
  dsize /= 2;
  if (dsize % group_size > 0) {
    dsize = dsize / group_size + 1;
  }
  else
  {
    dsize = dsize / group_size;
  }
  glm::uvec2 dispatch_size = {dsize, 1};
  if (arr_size % group_size > 0) dispatch_size.x++;
  
	compute_shader.dispatch(dispatch_size);
	compute_shader.wait();
	unsigned int collection_size = arr_size;
	std::vector<float> compute_data(collection_size);
	compute_shader.retrieve_ssbo(ssbo, compute_data.data(), collection_size * sizeof(float));
	debug_log("%f", compute_data[0]);
  //
  best_score = best.score;
  return AnimationIndex(dataBase, best.clip, best.frame);
}