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
  vec4 nodes[(uint)AnimationFeaturesNode::Count];
  vec4 nodesVelocity[(uint)AnimationFeaturesNode::Count];
  vec4 points[(uint)AnimationTrajectory::PathLength];
  vec4 pointsVelocity[(uint)AnimationTrajectory::PathLength];
  float angularVelocity[(uint)AnimationTrajectory::PathLength];
  float goalPathMatchingWeight;
  uint64_t tag;
  //uint padding[];
};

struct InOutBuffer
{
  vec4 nodes[(uint)AnimationFeaturesNode::Count];
  vec4 nodesVelocity[(uint)AnimationFeaturesNode::Count];
  vec4 points[(uint)AnimationTrajectory::PathLength];
  vec4 pointsVelocity[(uint)AnimationTrajectory::PathLength];
  float angularVelocity[(uint)AnimationTrajectory::PathLength];
  /*
  uint64_t tag;
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  */
};

struct ShaderMatchingScores
{
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint padding[2];
};

void store_database(AnimationDataBasePtr dataBase, const MotionMatchingSettings &mmsettings, uint feature_ssbo, int &size)
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
      nextFeatureCell.goalPathMatchingWeight = mmsettings.goalPathMatchingWeight;
      nextFeatureCell.tag = clip.tags.tags;
      featureData.push_back(nextFeatureCell);
      size++;
    }
  }
  store_ssbo(feature_ssbo, featureData.data(), sizeof(FeatureCell) * featuresCounter);
}

void store_goal_feature(const AnimationGoal& goal, const MotionMatchingSettings &mmsettings, uint uboBlock)
{
  float poseWeight = mmsettings.realism * mmsettings.poseMatchingWeight;
  float velocityWeight = mmsettings.realism * mmsettings.velocityMatchingWeight;
  InOutBuffer goal_feature;
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
  /*
  goal_feature.tag = goal.tags.tags;
  goal_feature.pose = 0;
  goal_feature.goal_tag = 0;
  goal_feature.goal_path = 0;
  goal_feature.trajectory_v = 0;
  goal_feature.trajectory_w = 0;
  goal_feature.full_score = 0;
  */
  glBindBuffer(GL_UNIFORM_BUFFER, uboBlock);
  glBindBufferBase(GL_UNIFORM_BUFFER, 2, uboBlock);
  glBufferData(GL_UNIFORM_BUFFER, 224, &goal_feature, GL_STREAM_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
 
  //debug_log("%f", goal_feature.angularVelocity[0]);
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
  static uint result_ssbo = 0;
  static int dataSize;
  static bool database_stored = false;
  static ShaderMatchingScores *scores;
  static uint uboBlock;

  if (!database_stored) 
  {
    feature_ssbo = create_ssbo(0);
    result_ssbo = create_ssbo(1);
    store_database(dataBase, mmsettings, feature_ssbo, dataSize);
    
    glGenBuffers(1, &uboBlock);
    glBindBufferBase(GL_UNIFORM_BUFFER, 2, uboBlock); 

    scores = new ShaderMatchingScores[dataSize];
    store_ssbo(result_ssbo, NULL, dataSize * sizeof(ShaderMatchingScores));

    database_stored = true;
  }
  //-------------------------------------------------/

  uint group_size = 256;
  

  auto compute_shader = get_compute_shader("compute_motion");
  compute_shader.use();
  store_goal_feature(goal, mmsettings, uboBlock);

  glm::uvec2 dispatch_size = {8, 1};
  
  uint invocations = dispatch_size.x * dispatch_size.y * group_size;

  int iterations = dataSize / invocations;
  if (dataSize % invocations > 0) iterations++;
  compute_shader.set_int("arr_size", dataSize);
  compute_shader.set_int("iterations", iterations);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, feature_ssbo);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, result_ssbo);
  glBindBufferBase(GL_UNIFORM_BUFFER, 2, uboBlock);
	compute_shader.dispatch(dispatch_size);
  auto sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glWaitSync(sync, 0, 1000000000);
	compute_shader.wait();
	retrieve_ssbo(result_ssbo, scores, dataSize * sizeof(ShaderMatchingScores), 1);
  
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
      if (idx % 100 == 0)
          debug_log("%f %f", scores[idx].full_score, cur.score.full_score);
      idx++;
    }
  }

  /*
  // parallel reduction for finding the minimum value in an array of positive floats
  uint arr_size = 256;
  static uint a = 0;
  GLfloat *data = new GLfloat[arr_size];
	for (uint i = 0; i < arr_size; ++i) {
		data[i] = i + 100;
  }
  data[a % 256] = 15;
  a++;
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
  dispatch_size = {dsize, 1};
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