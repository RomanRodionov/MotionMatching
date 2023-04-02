#include "Animation/animation_player.h"
#include "Animation/settings.h"
#include "Animation/MotionMatching/motion_matching.h"
#include <map>
#include "application/profile_tracker.h"
#include <ecs.h>
#include <camera.h>
#include <render/material.h>
#include <profiler/profiler.h>
#include "microprofile/microprofile.h"
#include "Animation/mm_cs_structures.h"


SYSTEM(stage=act;after=motion_matching_cs_update, motion_matching_update) init_cs_data(
  Asset<AnimationDataBase> &dataBase,
  bool &mm_mngr,
  int &groups_per_char,
  int &group_size,
  CSData &cs_data,
  SettingsContainer &settingsContainer,
  int *mmIndex)
{
  if (!cs_data.initialized)
  {
    cs_data.dispatch_size = groups_per_char;
    int max_dispatch;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &max_dispatch);
    if (max_dispatch < cs_data.dispatch_size)
    {
      cs_data.dispatch_size = max_dispatch;
    }
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &cs_data.max_parallel_char);
    if (cs_data.max_parallel_char > MAX_QUEUE_SIZE)
    {
      cs_data.max_parallel_char = MAX_QUEUE_SIZE;
    }
    cs_data.invocations = cs_data.dispatch_size * group_size;
    cs_data.group_size = group_size;
    const MotionMatchingSettings &mmsettings = settingsContainer.motionMatchingSettings[mmIndex ? *mmIndex : 0].second;
    cs_data.feature_ssbo = create_ssbo();
    store_database(dataBase, mmsettings, cs_data.clip_labels, cs_data.feature_ssbo, cs_data.dataSize);
    cs_data.iterations = cs_data.dataSize / cs_data.invocations;
    if (cs_data.dataSize % cs_data.invocations > 0) cs_data.iterations++;
    cs_data.resSize = cs_data.dispatch_size;
    cs_data.scores = new ShaderMatchingScores[MAX_QUEUE_SIZE * cs_data.resSize];
    cs_data.buffers_parity.init_size(MAX_QUEUE_SIZE * sizeof(FeatureCell), MAX_QUEUE_SIZE * cs_data.resSize * sizeof(ShaderMatchingScores));
    cs_data.buffers_parity.create_pair();
    cs_data.initialized = true;
    debug_log("cs data have been initialized");
  }
}

SYSTEM(stage=act;before=animation_player_update) motion_matching_cs_update(
  Asset<AnimationDataBase> &dataBase,
  bool &mm_mngr,
  GoalsBuffer &goal_buffer,
  ResultsBuffer &result_buffer,
  CSData &cs_data,
  SettingsContainer &settingsContainer,
  int *mmIndex)
{
  if (goal_buffer.get_size() > 0)
  {
    MICROPROFILE_SCOPEI("MM_CS_UPDATE", "mm_cs_update", 0xff0f0f);
    const MotionMatchingSettings &mmsettings = settingsContainer.motionMatchingSettings[mmIndex ? *mmIndex : 0].second;
    auto compute_shader = get_compute_shader("compute_motion");
    compute_shader.use();
    compute_shader.set_int("data_size", cs_data.dataSize);
    compute_shader.set_int("iterations", cs_data.iterations);
    compute_shader.set_float("goalPathMatchingWeight", mmsettings.goalPathMatchingWeight);
    compute_shader.set_float("realism", mmsettings.realism);
    vector<FeatureCell> featureData;
    //debug_log("%d\n", goal_buffer.get_size());
    uint step = 0;
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cs_data.feature_ssbo);
    while(goal_buffer.ready())
    {        
      if (cs_data.buffers_parity.ready(step))
      {
        if (cs_data.goals.size() <= step)
        {
          cs_data.goals.push_back({});
        }
        else{
          cs_data.goals[step].clear();
        }
        uint queue_size = goal_buffer.get_size() < cs_data.max_parallel_char ? goal_buffer.get_size() : cs_data.max_parallel_char;
        cs_data.buffers_parity.bind_pair(step, 2, 1);
        compute_shader.set_int("queue_size", queue_size);
        featureData.clear();
        for (uint idx = 0; idx < queue_size; ++idx)
        {
          IdentifiedGoal goal = goal_buffer.get();
          cs_data.goals[step].push_back(goal);
          FeatureCell goal_feature;
          pack_goal_feature(goal.goal, mmsettings, goal_feature);
          featureData.push_back(goal_feature);
        }
        {
          ProfilerLabelGPU label("store goals");
          MICROPROFILE_SCOPEGPUI("store_goals", 0xff0f0f);
          cs_data.buffers_parity.store_goal(featureData.data(), sizeof(FeatureCell), featureData.size());
        }
        {
          ProfilerLabelGPU label("dispatch cs");
          MICROPROFILE_SCOPEGPUI("mm_shader", 0xff0f0f);
          compute_shader.dispatch(cs_data.dispatch_size, queue_size);
        }
        cs_data.buffers_parity.set_sync();
        //compute_shader.wait();
      }
      step++;
    }
  }
}

SYSTEM(stage=animation;) motion_matching_cs_retrieve(
  Asset<AnimationDataBase> &dataBase,
  bool &mm_mngr,
  GoalsBuffer &goal_buffer,
  ResultsBuffer &result_buffer,
  CSData &cs_data,
  SettingsContainer &settingsContainer,
  int *mmIndex)
{
  if (cs_data.goals.size() > 0)
  {
    MICROPROFILE_SCOPEI("MM_CS_RETRIEVE", "mm_cs_retrieve_function", 0xff1f1f);
    //std::this_thread::sleep_for(std::chrono::nanoseconds(100000000));
    const MotionMatchingSettings &mmsettings = settingsContainer.motionMatchingSettings[mmIndex ? *mmIndex : 0].second;
    auto compute_shader = get_compute_shader("compute_motion");
    compute_shader.use();
    uint step = 0;
    for (auto goals : cs_data.goals)
    {        
      if (goals.size() > 0 & cs_data.buffers_parity.wait_sync(step, 500000))
      {
        uint queue_size;
        cs_data.buffers_parity.set_index(step);
        {
          ProfilerLabelGPU label("retrieve cs");
          MICROPROFILE_SCOPEGPUI("retrieve_mm_from_shader", 0xff0f0f);
          MICROPROFILE_SCOPEI("MM_CS_RETRIEVE", "mm_cs_retrieve", 0xff0f0f);
          cs_data.buffers_parity.retrieve_result(cs_data.scores, cs_data.resSize * sizeof(ShaderMatchingScores), queue_size);
        }
        ShaderMatchingScores best_matching;
        {
          MICROPROFILE_SCOPEI("MM_CS_UPDATE", "frame_search", 0xff0f0f);
          for (uint idx = 0; idx < queue_size; ++idx)
          {
            best_matching = cs_data.scores[cs_data.resSize * idx];
            for (int i = 1; i < cs_data.resSize; ++i)
            {
              if (cs_data.scores[cs_data.resSize * idx + i].full_score < best_matching.full_score)
              {
                best_matching = cs_data.scores[cs_data.resSize * idx + i];
              }
            }
            uint l_border = 0, r_border = cs_data.clip_labels.size() - 1;
            uint clip_idx = r_border / 2;
            while (l_border != r_border)
            {
              if (best_matching.idx >= cs_data.clip_labels[r_border])
              {
                l_border = r_border;
              }
              else if (best_matching.idx < cs_data.clip_labels[clip_idx])
              {
                r_border = clip_idx;
                clip_idx = (r_border + l_border) / 2;
              }
              else if (best_matching.idx < cs_data.clip_labels[clip_idx + 1])
              {
                l_border = r_border = clip_idx;
              }
              else
              {
                l_border = clip_idx;
                clip_idx = (r_border + l_border + 1) / 2;
              }
            }

            if ((best_matching.idx < cs_data.clip_labels[clip_idx] + dataBase->clips[clip_idx].duration) &&
                (has_goal_tags(goals[idx].goal.tags, dataBase->clips[clip_idx].tags))){
              goals[idx].best_score.full_score = best_matching.full_score;
              goals[idx].best_score.goal_path = best_matching.goal_path;
              goals[idx].best_score.pose = best_matching.pose;
              goals[idx].best_score.trajectory_v = best_matching.trajectory_v;
              goals[idx].best_score.trajectory_w = best_matching.trajectory_w;
              result_buffer.push(AnimationIndex(dataBase, clip_idx, best_matching.idx - cs_data.clip_labels[clip_idx]), 
                  goals[idx].best_score, goals[idx].charId);
            }
            else
            {
              result_buffer.push(AnimationIndex(dataBase, goals[idx].curClip, goals[idx].curCadr), 
                  goals[idx].best_score, goals[idx].charId);
            }
          }
        }
      }
      step++;
    }
  }
}