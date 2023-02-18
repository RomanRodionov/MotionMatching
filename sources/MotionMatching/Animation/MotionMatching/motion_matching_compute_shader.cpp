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
  store_ssbo(feature_ssbo, featureData.data(), sizeof(FeatureCell) * featuresCounter);
}

void store_goal_feature(const AnimationGoal& goal, const MotionMatchingSettings &mmsettings, uint uboBlock)
{
  float poseWeight = mmsettings.poseMatchingWeight;
  float velocityWeight = mmsettings.velocityMatchingWeight;
  FeatureCell goal_feature;
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
  store_ubo(uboBlock, &goal_feature, sizeof(FeatureCell));
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
  static uint feature_ssbo = create_ssbo(0);
  static uint result_ssbo = create_ssbo(1);
  static int dataSize, resSize;
  static bool database_stored = false;
  static ShaderMatchingScores *scores;
  static uint uboBlock = create_ubo(2);
  static uint group_size = 256;
  static glm::uvec2 dispatch_size = {4, 1};
  static uint invocations = dispatch_size.x * dispatch_size.y * group_size;
  static int iterations;
  static vector<uint> clip_labels;

  if (!database_stored) 
  {
    store_database(dataBase, mmsettings, clip_labels, feature_ssbo, dataSize);
    iterations = dataSize / invocations;
    if (dataSize % invocations > 0) iterations++;
    resSize = dataSize / (iterations * group_size);
    if (dataSize % (iterations * group_size) > 0) resSize++;
    scores = new ShaderMatchingScores[resSize];
    store_ssbo(result_ssbo, NULL, resSize * sizeof(ShaderMatchingScores));
    database_stored = true;
  }
  //-------------------------------------------------/

  auto compute_shader = get_compute_shader("compute_motion");
  compute_shader.use();
  store_goal_feature(goal, mmsettings, uboBlock);

  compute_shader.set_int("data_size", dataSize);
  compute_shader.set_int("iterations", iterations);
  compute_shader.set_float("goalPathMatchingWeight", mmsettings.goalPathMatchingWeight);
  compute_shader.set_float("realism", mmsettings.realism);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, feature_ssbo);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, result_ssbo);
  glBindBufferBase(GL_UNIFORM_BUFFER, 2, uboBlock);
  
  {
    ProfilerLabelGPU label("dispatch cs");
	  compute_shader.dispatch(dispatch_size);
	  compute_shader.wait();
	  retrieve_ssbo(result_ssbo, scores, resSize * sizeof(ShaderMatchingScores));
  }

  ShaderMatchingScores best_matching;
  for (int i = 0; i < resSize; ++i)
  {
    if (i == 0 || scores[i].full_score < best_matching.full_score)
    {
      best_matching = scores[i];
    }
  }
  
  uint l_border = 0, r_border = clip_labels.size() - 1;
  uint clip_idx = r_border / 2;
  while (l_border != r_border)
  {
    if (best_matching.idx >= clip_labels[r_border])
    {
      l_border = r_border;
    }
    else if (best_matching.idx < clip_labels[clip_idx])
    {
      r_border = clip_idx;
      clip_idx = (r_border + l_border) / 2;
    }
    else if (best_matching.idx < clip_labels[clip_idx + 1])
    {
      l_border = r_border = clip_idx;
    }
    else
    {
      l_border = clip_idx;
      clip_idx = (r_border + l_border + 1) / 2;
    }
  }

  if ((best_matching.idx < clip_labels[clip_idx] + dataBase->clips[clip_idx].duration) &&
      (has_goal_tags(goal.tags, dataBase->clips[clip_idx].tags))){
    best_score.full_score = best_matching.full_score;
    best_score.goal_path = best_matching.goal_path;
    best_score.pose = best_matching.pose;
    best_score.trajectory_v = best_matching.trajectory_v;
    best_score.trajectory_w = best_matching.trajectory_w;
    return AnimationIndex(dataBase, clip_idx, best_matching.idx - clip_labels[clip_idx]);
  }
  return AnimationIndex(dataBase, curClip, curCadr);
}