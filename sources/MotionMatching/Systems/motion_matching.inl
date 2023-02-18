#include "Animation/animation_player.h"
#include "Animation/settings.h"
#include "Animation/MotionMatching/motion_matching.h"
#include <map>
#include "application/profile_tracker.h"
#include <ecs.h>
#include <camera.h>
#include <render/material.h>

AnimationIndex solve_motion_matching(
  AnimationDataBasePtr dataBase,
  const AnimationIndex &index,
  const AnimationGoal &goal,
  MatchingScores &best_score,
  const MotionMatchingSettings &mmsettings);

AnimationIndex solve_motion_matching_cs(
  AnimationDataBasePtr dataBase,
  const AnimationIndex &index,
  const AnimationGoal &goal,
  MatchingScores &best_score,
  const MotionMatchingSettings &mmsettings);

void get_motion_matching_statistic(
  AnimationDataBasePtr dataBase,
  const AnimationIndex &index,
  const AnimationGoal &goal,
  const MotionMatchingSettings &mmsettings);


AnimationIndex solve_motion_matching_vp_tree(
  AnimationDataBasePtr dataBase,
  const AnimationGoal &goal,
  float tolerance_erorr);
  
AnimationIndex solve_motion_matching_cover_tree(
  AnimationDataBasePtr dataBase,
  const AnimationGoal &goal,
  float tolerance_erorr);

AnimationIndex solve_motion_matching_kd_tree(
  AnimationDataBasePtr dataBase,
  const AnimationGoal &goal,
  float tolerance_error);

MotionMatching::MotionMatching(AnimationDataBasePtr dataBase, AnimationLerpedIndex index):
dataBase(dataBase), index(index), skip_time(0), lod(0)
{  
}

AnimationLerpedIndex MotionMatching::get_index() const
{
  return index;
}

static bool trajection_tolerance_test(AnimationIndex index, const AnimationGoal &goal, const MotionMatchingSettings &mmsettings, float pathErrorTolerance)
{
  const AnimationClip &clip = index.get_clip();
  int frame = index.get_cadr_index();
  if (has_goal_tags(goal.tags, clip.tags))
  {
    const AnimationTrajectory &trajectory = clip.features[frame].trajectory;
    float path_cost = path_norma(trajectory, goal.feature.trajectory, mmsettings);
    return path_cost < pathErrorTolerance;
  }
  return false;
}

struct IdentifiedGoal
{
  AnimationGoal goal;
  uint curClip, curCadr;
  MatchingScores best_score;
  ecs::EntityId eid;
};

constexpr int MIN_QUEUE_SIZE = 1;
constexpr float WAIT_LIMIT = 0.001f;
struct GoalsBuffer : ecs::Singleton
{
  std::queue<IdentifiedGoal> goals = {};
  float wait_time = 0;
  uint get_size()
  {
    return goals.size();
  }
  bool ready()
  {
    return (goals.size() >= MIN_QUEUE_SIZE) || ((Time::time() - wait_time) > WAIT_LIMIT && goals.size() > 0);
  }
  void push(AnimationGoal goal, uint curClip, uint curCadr, MatchingScores best_score, ecs::EntityId eid)
  {
    if (goals.size() % MIN_QUEUE_SIZE == 0)
      wait_time = Time::time();
    goals.push({goal, curClip, curCadr, best_score, eid});
  }
  IdentifiedGoal get()
  {
    IdentifiedGoal front_element = goals.front();
    goals.pop();
    return front_element;
  }
};

struct ResultsBuffer : ecs::Singleton
{
  std::map<ecs::EntityId, std::queue<AnimationIndex>> results;
  std::map<ecs::EntityId, std::queue<MatchingScores>> scores;
  bool ready(ecs::EntityId eid)
  {
    return results[eid].size() > 0;
  }
  void push(AnimationIndex result, MatchingScores best_score, ecs::EntityId eid)
  {
    results[eid].push(result);
    scores[eid].push(best_score);
  }
  AnimationIndex get(ecs::EntityId eid, MatchingScores &best_score)
  {
    AnimationIndex front_element = results[eid].front();
    results[eid].pop();
    best_score = scores[eid].front();
    scores[eid].pop();
    return front_element;
  }
};

struct CSData : ecs::Singleton
{
  uint feature_ssbo, result_ssbo, goal_ubo, group_size = 256;
  glm::uvec2 dispatch_size = {4, 1};
  uint invocations = dispatch_size.x * dispatch_size.y * group_size;
  vector<uint> clip_labels;
  int dataSize, resSize, iterations;
  ShaderMatchingScores *scores;
  bool initialized = false;
};

SYSTEM(stage=act;before=motion_matching_cs_update, motion_matching_update) init_cs_data(
  Asset<AnimationDataBase> &dataBase,
  bool &mm_mngr,
  CSData &cs_data,
  SettingsContainer &settingsContainer,
  int *mmIndex)
{
  if (!cs_data.initialized)
  {
    const MotionMatchingSettings &mmsettings = settingsContainer.motionMatchingSettings[mmIndex ? *mmIndex : 0].second;
    cs_data.feature_ssbo = create_ssbo(0);
    cs_data.result_ssbo = create_ssbo(1);
    cs_data.goal_ubo = create_ubo(2);
    store_database(dataBase, mmsettings, cs_data.clip_labels, cs_data.feature_ssbo, cs_data.dataSize);
    cs_data.iterations = cs_data.dataSize / cs_data.invocations;
    if (cs_data.dataSize % cs_data.invocations > 0) cs_data.iterations++;
    cs_data.resSize = cs_data.dataSize / (cs_data.iterations * cs_data.group_size);
    if (cs_data.dataSize % (cs_data.iterations * cs_data.group_size) > 0) cs_data.resSize++;
    cs_data.scores = new ShaderMatchingScores[cs_data.resSize];
    store_ssbo(cs_data.result_ssbo, NULL, cs_data.resSize * sizeof(ShaderMatchingScores));
    cs_data.initialized = true;
    debug_log("cs data have been initialized");
  }
}

SYSTEM(stage=act;before=animation_player_update;after=motion_matching_update) motion_matching_cs_update(
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
    const MotionMatchingSettings &mmsettings = settingsContainer.motionMatchingSettings[mmIndex ? *mmIndex : 0].second;
    auto compute_shader = get_compute_shader("compute_motion");
    compute_shader.use();
    compute_shader.set_int("data_size", cs_data.dataSize);
    compute_shader.set_int("iterations", cs_data.iterations);
    compute_shader.set_float("goalPathMatchingWeight", mmsettings.goalPathMatchingWeight);
    compute_shader.set_float("realism", mmsettings.realism);
    vector<IdentifiedGoal> goals;
    while(goal_buffer.ready())
    {
      uint queue_size = goal_buffer.get_size();
      for (uint idx = 0; idx < MIN_QUEUE_SIZE && idx < queue_size; ++idx)
      {
        IdentifiedGoal goal = goal_buffer.get();
        goals.push_back(goal);
        store_goal_feature(goal.goal, mmsettings, cs_data.goal_ubo);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cs_data.feature_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cs_data.result_ssbo);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, cs_data.goal_ubo);

        compute_shader.dispatch(cs_data.dispatch_size);
        compute_shader.wait();
      }
      retrieve_ssbo(cs_data.result_ssbo, cs_data.scores, cs_data.resSize * sizeof(ShaderMatchingScores));
      ShaderMatchingScores best_matching;
      for (uint idx = 0; idx < MIN_QUEUE_SIZE && idx < queue_size; ++idx)
      {
        best_matching = cs_data.scores[0];
        for (int i = 1; i < cs_data.resSize; ++i)
        {
          if (cs_data.scores[i].full_score < best_matching.full_score)
          {
            best_matching = cs_data.scores[i];
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
              goals[idx].best_score, goals[idx].eid);
        }
        else
        {
          result_buffer.push(AnimationIndex(dataBase, goals[idx].curClip, goals[idx].curCadr), 
              goals[idx].best_score, goals[idx].eid);
        }
      }
      goals.clear();
    }
  }
}

constexpr int MAX_SAMPLES = 10000;
struct MMProfiler : ecs::Singleton
{
  //vector<ProfileTracker> trackers;
  vector<ProfileTracker> avgTrackers;
  //Tag tagsCount;
  bool stopped = false;
  bool inited = false;
  std::vector<int> counter;
  MMProfiler()=default;
  void init(const std::vector<std::pair<std::string, MotionMatchingOptimisationSettings>>&settings)
  {
    if (inited)
      return;
    inited = true;
    counter.resize(settings.size(), 0);

    for (const auto &[name, s] :settings)
    {
      avgTrackers.emplace_back(project_path("profile/" + name + ".csv"), MAX_SAMPLES);
    }
  }
  ProfileTracker &get_tracker(int solver)
  {
    return avgTrackers[solver];
  }
};

SYSTEM(stage=act;before=animation_player_update) motion_matching_update(
  Transform &transform,
  AnimationPlayer &animationPlayer,
  Asset<Material> &material,
  int *mmIndex,
  int &mmOptimisationIndex,
  bool updateMMStatistic,
  Settings &settings,
  SettingsContainer &settingsContainer,
  MMProfiler &profiler,
  const MainCamera &mainCamera,
  GoalsBuffer &goal_buffer,
  ResultsBuffer &result_buffer,
  ecs::EntityId &eid)
{
  float dt = Time::delta_time();
  
  if (animationPlayer.playerType ==  AnimationPlayerType::MotionMatching)
  {
    profiler.init(settingsContainer.motionMatchingOptimisationSettings);
    MotionMatching &matching = animationPlayer.motionMatching;
    const MotionMatchingSettings &mmsettings = settingsContainer.motionMatchingSettings[mmIndex ? *mmIndex : 0].second;
    const MotionMatchingOptimisationSettings &OptimisationSettings = 
      settingsContainer.motionMatchingOptimisationSettings[mmOptimisationIndex].second;
    float dist = length(mainCamera.position - transform.get_position());

    int j = 0;
    vec3 lodColor(0.f);
    vec3 lodColors[] = {vec3(0), vec3(0,1,0), vec3(0,0,1), vec3(1,0,0)};
    for (; j < 3; j++)
      if (dist < OptimisationSettings.lodDistances[j])
        break;
    lodColor = lodColors[j];
    matching.lod = j;
    
    material->set_property("material.AdditionalColor", lodColor);


    auto &index = matching.index;
    AnimationIndex saveIndex = index.current_index();
    index.update(dt, settings.lerpTime);
    matching.skip_time += dt;
    

    AnimationIndex currentIndex = index.current_index();
    if (saveIndex != currentIndex)
    {
      ProfileTrack track;
      auto &goal = animationPlayer.inputGoal;
      auto dataBase = matching.dataBase;
      settings.TotalMMCount++;
      matching.lod = OptimisationSettings.lodOptimisation ? matching.lod : 0;
      float lodSkipTime = OptimisationSettings.lodSkipSeconds[matching.lod];
      if (!OptimisationSettings.lodOptimisation || matching.skip_time >= lodSkipTime)
      {
        settings.afterLodOptimization++;
        matching.skip_time -= lodSkipTime;
        if (!OptimisationSettings.lodOptimisation)
          matching.skip_time = 0;
        if (OptimisationSettings.trajectoryErrorToleranceTest &&
            trajection_tolerance_test(currentIndex, goal, mmsettings, OptimisationSettings.pathErrorTolerance))
        {
          goto afterMM;
        }
        settings.afterTrajectoryToleranceTest++;
        goal.feature.features = currentIndex.get_feature();
        if (updateMMStatistic)
        {
          get_motion_matching_statistic(dataBase, currentIndex, goal, mmsettings);
        }
        matching.bestScore = {0,0,0,0,0,0};
        AnimationIndex best_index;

        switch ((MotionMatchingSolverType)OptimisationSettings.solverType)
        {
        default: 
        case MotionMatchingSolverType::BruteForce :
          best_index = solve_motion_matching(dataBase, currentIndex, goal, matching.bestScore, mmsettings);
          break;
        case MotionMatchingSolverType::CSBruteForce :
          best_index = solve_motion_matching_cs(dataBase, currentIndex, goal, matching.bestScore, mmsettings);
          break;
        case MotionMatchingSolverType::VPTree :
          best_index = solve_motion_matching_vp_tree(dataBase, goal, OptimisationSettings.vpTreeErrorTolerance);
          break;
        case MotionMatchingSolverType::CoverTree :
          best_index = solve_motion_matching_cover_tree(dataBase, goal, OptimisationSettings.vpTreeErrorTolerance);
          break;
        case MotionMatchingSolverType::KDTree :
          best_index = solve_motion_matching_kd_tree(dataBase, goal, OptimisationSettings.vpTreeErrorTolerance);
          break;
        }
        bool can_jump = true;
        for (const AnimationIndex &index : index.get_indexes())
          can_jump &= AnimationIndex::can_jump(index, best_index);
        if (can_jump)
        {
          index.play_lerped(best_index, settings.maxLerpIndex);
        }
      }
afterMM:
      if (settings.startTesting)
      {
        float delta = track.delta();
        auto &tracker = profiler.get_tracker(mmOptimisationIndex);
        tracker.update(delta);
        if (tracker.was_stopped())
        {
          mmOptimisationIndex++;
          if (mmOptimisationIndex >= (int)settingsContainer.motionMatchingOptimisationSettings.size())
          {
            settings.startTesting = false;
            ofstream os(project_path("profile/average.txt"));
            os << "name;average;max;avg_ratio;max_ratio\n";
            const auto &tracker = profiler.get_tracker(0);
            float avgBruteForce = tracker.averageTime;
            float maxBruteForce = tracker.maxTime;
            for (int i = 0; i < mmOptimisationIndex; i++)
            {
              const auto &tracker = profiler.get_tracker(i);
              os << settingsContainer.motionMatchingOptimisationSettings[i].first
               << ";" << tracker.averageTime << ";" << tracker.maxTime 
               << ";" << avgBruteForce / tracker.averageTime << ";" << maxBruteForce/ tracker.maxTime 
               << '\n';

            }
            debug_log("profiling finished");
          }
        }
      }
    }
    animationPlayer.index = matching.get_index();
  }
}

AnimationDataBasePtr MotionMatching::get_data_base() const
{
  return dataBase;
}