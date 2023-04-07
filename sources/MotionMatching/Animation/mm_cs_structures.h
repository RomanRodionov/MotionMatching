#include "MotionMatching/motion_matching.h"
#include <map>
#include "application/profile_tracker.h"
#include <ecs.h>
#include <camera.h>
#include <render/material.h>
#include <profiler/profiler.h>
#include "microprofile/microprofile.h"

struct IdentifiedGoal
{
  AnimationGoal goal;
  uint curClip, curCadr;
  MatchingScores best_score;
  int charId;
};

constexpr int MAX_QUEUE_SIZE = 5000;
struct GoalsBuffer : ecs::Singleton
{
  std::queue<IdentifiedGoal> goals = {};
  uint get_size();
  bool ready();
  void push(AnimationGoal goal, uint curClip, uint curCadr, MatchingScores best_score, int charId);
  IdentifiedGoal get();
};

struct ResultsBuffer : ecs::Singleton
{
  std::map<int, AnimationIndex> results;
  std::map<int, MatchingScores> scores;
  bool ready(int charId);
  void push(AnimationIndex result, MatchingScores best_score, int charId);
  AnimationIndex get(int charId, MatchingScores &best_score);
};

class BuffersParity
{
  vector<uint> goal_ssbo, result_ssbo, occupancy;
  vector<GLsync> sync;
  uint goal_init_size, result_init_size, index;
public:
  BuffersParity();
  void init_size(uint goal_size, uint result_size);
  void create_pair();
  bool ready(uint idx);
  uint get_size();
  uint get_goal();
  uint get_result();
  void store_goal(void *data, uint item_size, uint item_count);
  void retrieve_result(void *data, uint item_size, uint &item_count);
  void bind_pair(uint idx, uint goal_binding, uint result_binding);
  void set_index(uint idx);
  void set_sync();
  bool wait_sync(uint idx, GLuint64 time_out);
};

struct CSData : ecs::Singleton
{
  uint feature_ssbo, group_size;
  BuffersParity buffers_parity;
  int dispatch_size;
  int max_parallel_char;
  uint invocations;
  vector<uint> clip_labels;
  int dataSize, resSize, iterations;
  ShaderMatchingScores *scores;
  bool initialized = false;
  vector<vector<IdentifiedGoal>> goals;
};

void push_motion_matching_cs_task(
  const AnimationIndex &index,
  const AnimationGoal &goal,
  MatchingScores &best_score,
  int& charId,
  GoalsBuffer& goal_buffer);

  
AnimationIndex get_motion_matching_cs_results(
  AnimationDataBasePtr dataBase,
  const AnimationIndex &index,
  MatchingScores &best_score,
  int& charId,
  GoalsBuffer& goal_buffer,
  ResultsBuffer& result_buffer);