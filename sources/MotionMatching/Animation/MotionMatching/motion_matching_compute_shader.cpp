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

struct FeatureCell
{
  vec3 nodes[(uint)AnimationFeaturesNode::Count];
  vec3 nodesVelocity[(uint)AnimationFeaturesNode::Count];
  vec3 points[(uint)AnimationTrajectory::PathLength];
  vec3 pointsVelocity[(uint)AnimationTrajectory::PathLength];
  float angularVelocity[(uint)AnimationTrajectory::PathLength];
  float goalPathMatchingWeight;
  uint64_t tag;
  //uint padding[];
};

struct InOutBuffer
{
  vec3 nodes[(uint)AnimationFeaturesNode::Count];
  vec3 nodesVelocity[(uint)AnimationFeaturesNode::Count];
  vec3 points[(uint)AnimationTrajectory::PathLength];
  vec3 pointsVelocity[(uint)AnimationTrajectory::PathLength];
  float angularVelocity[(uint)AnimationTrajectory::PathLength];
  uint64_t tag;
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint padding[3];
};

struct ShaderMatchingScores
{
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint padding[2];
};

void store_database(AnimationDataBasePtr dataBase, const MotionMatchingSettings &mmsettings, uint feature_ssbo, uint &size)
{
  float poseWeight = mmsettings.realism * mmsettings.poseMatchingWeight;
  float velocityWeight = mmsettings.realism * mmsettings.velocityMatchingWeight;
  std::vector<FeatureCell> featureData;
  uint featuresCounter = 0;
  size = 0;
  for (uint nextClip = 0; nextClip < dataBase->clips.size(); nextClip++)
  {
    const AnimationClip &clip = dataBase->clips[nextClip];
    featuresCounter += clip.duration;
    for (uint nextCadr = 0, n = clip.duration; nextCadr < n; nextCadr++)
    {
      auto &frame = clip.features[nextCadr];
      FeatureCell nextFeatureCell;
      for (uint node = 0; node < (uint)AnimationFeaturesNode::Count; node++)
      {
        nextFeatureCell.nodes[node] = frame.features.nodes[node] * float(mmsettings.nodeWeights[node]) * poseWeight;
        if (mmsettings.velocityMatching)
          nextFeatureCell.nodesVelocity[node] = frame.features.nodesVelocity[node] * float(mmsettings.velocitiesWeights[node]) * velocityWeight;
        else
          nextFeatureCell.nodesVelocity[node] = glm::vec3(0, 0, 0);
      }
      for (uint point = 0; point < (uint)AnimationTrajectory::PathLength; point++)
      {
        nextFeatureCell.points[point] = frame.trajectory.trajectory[point].point;
        nextFeatureCell.pointsVelocity[point] = frame.trajectory.trajectory[point].velocity * mmsettings.goalVelocityWeight;
        nextFeatureCell.angularVelocity[point] = frame.trajectory.trajectory[point].angularVelocity * mmsettings.goalAngularVelocityWeight;
      }
      nextFeatureCell.goalPathMatchingWeight = mmsettings.goalPathMatchingWeight;
      nextFeatureCell.tag = clip.tags.tags;
      featureData.push_back(nextFeatureCell);
      size++;
    }
  }
  store_ssbo(feature_ssbo, featureData.data(), sizeof(FeatureCell) * featuresCounter);
}

void store_goal_feature(const AnimationGoal& goal, const MotionMatchingSettings &mmsettings, uint feature_ssbo)
{
  float poseWeight = mmsettings.realism * mmsettings.poseMatchingWeight;
  float velocityWeight = mmsettings.realism * mmsettings.velocityMatchingWeight;
  InOutBuffer goal_feature;
  for (uint node = 0; node < (uint)AnimationFeaturesNode::Count; node++)
  {
    goal_feature.nodes[node] = goal.feature.features.nodes[node] * float(mmsettings.nodeWeights[node]) * poseWeight;
    if (mmsettings.velocityMatching)
      goal_feature.nodesVelocity[node] = goal.feature.features.nodesVelocity[node] * float(mmsettings.velocitiesWeights[node]) * velocityWeight;
    else
      goal_feature.nodesVelocity[node] = glm::vec3(0, 0, 0);
  }
  for (uint point = 0; point < (uint)AnimationTrajectory::PathLength; point++)
  {
    goal_feature.points[point] = goal.feature.trajectory.trajectory[point].point;
    goal_feature.pointsVelocity[point] = goal.feature.trajectory.trajectory[point].velocity * mmsettings.goalVelocityWeight;
    goal_feature.angularVelocity[point] = goal.feature.trajectory.trajectory[point].angularVelocity * mmsettings.goalAngularVelocityWeight;
  }
  goal_feature.tag = goal.tags.tags;
  goal_feature.pose = 0;
  goal_feature.goal_tag = 0;
  goal_feature.goal_path = 0;
  goal_feature.trajectory_v = 0;
  goal_feature.trajectory_w = 0;
  goal_feature.full_score = 0;
  store_ssbo(feature_ssbo, &goal_feature, sizeof(InOutBuffer));
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

  // temporarily here, then I will take it somewhere \/
  static uint feature_ssbo = 0;
  static uint goal_feature_ssbo = 0;
  static uint result_ssbo = 0;
  static uint dataSize;
  static bool database_stored = false;
  static ShaderMatchingScores *scores;

  if (!database_stored) 
  {
    feature_ssbo = create_ssbo(1);
    goal_feature_ssbo = create_ssbo(2);
    result_ssbo = create_ssbo(3);
    store_database(dataBase, mmsettings, feature_ssbo, dataSize);

    scores = new ShaderMatchingScores[dataSize];
    store_ssbo(result_ssbo, scores, dataSize * sizeof(ShaderMatchingScores));

    database_stored = true;
  }
  //-------------------------------------------------/

  store_goal_feature(goal, mmsettings, goal_feature_ssbo);

  uint group_size = 256;
   
  auto compute_shader = get_compute_shader("compute_motion");
  compute_shader.use();

  glm::uvec2 dispatch_size = {8, 1};
  
  uint invocations = dispatch_size.x * dispatch_size.y * group_size;

  uint iterations = dataSize / invocations;
  if (dataSize % invocations > 0) iterations++;

  compute_shader.set_int("arr_size", dataSize);
  compute_shader.set_int("iterations", iterations);

	compute_shader.dispatch(dispatch_size);
  auto sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glWaitSync(sync, 0, 100000000000);
	compute_shader.wait();
  
	retrieve_ssbo(result_ssbo, scores, dataSize * sizeof(ShaderMatchingScores));

  ArgMin best = {INFINITY, curClip, curCadr, best_score};
  
  uint idx = 0;

  for (uint nextClip = 0; nextClip < dataBase->clips.size(); nextClip++)
  {
    const AnimationClip &clip = dataBase->clips[nextClip];

    if (!has_goal_tags(goal.tags, clip.tags))
    { 
      idx += clip.duration;
      continue;
    }
    for (uint nextCadr = 0, n = clip.duration; nextCadr < n; nextCadr++)
    {
      MatchingScores score = get_score(clip.features[nextCadr], goal.feature, mmsettings);
      
      float matching = score.full_score;
      ArgMin cur = {matching, nextClip, nextCadr, score};
      best = mm_min2(best, cur);
      debug_log("%f %f", scores[idx].full_score, cur.score.full_score);
      idx++;
    }
  }

  /*
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
  uint ssbo = create_ssbo();
  store_ssbo(ssbo, data, arr_size * sizeof(float));
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
	retrieve_ssbo(ssbo, compute_data.data(), collection_size * sizeof(float));
	debug_log("%f", compute_data[0]);
  */
  best_score = best.score;
  return AnimationIndex(dataBase, best.clip, best.frame);
}