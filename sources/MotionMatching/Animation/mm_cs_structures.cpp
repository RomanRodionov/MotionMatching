#include "MotionMatching/motion_matching.h"
#include <map>
#include "application/profile_tracker.h"
#include <ecs.h>
#include <camera.h>
#include <render/material.h>
#include <profiler/profiler.h>
#include "microprofile/microprofile.h"
#include "mm_cs_structures.h"


uint GoalsBuffer::get_size()
{
  return goals.size();
}

bool GoalsBuffer::ready()
{
  return goals.size() > 0;
}

void GoalsBuffer::push(AnimationGoal goal, uint curClip, uint curCadr, MatchingScores best_score, int charId)
{
  goals.push({goal, curClip, curCadr, best_score, charId});
}

IdentifiedGoal GoalsBuffer::get()
{
  IdentifiedGoal front_element = goals.front();
  goals.pop();
  return front_element;
}

bool ResultsBuffer::ready(int charId)
{
  return results.find(charId) != results.end();
}

void ResultsBuffer::push(AnimationIndex result, MatchingScores best_score, int charId)
{
  results[charId] = result;
  scores[charId] = best_score;
}

AnimationIndex ResultsBuffer::get(int charId, MatchingScores &best_score)
{
  AnimationIndex front_element = results[charId];
  results.erase(charId);
  best_score = scores[charId];
  scores.erase(charId);
  return front_element;
}

BuffersParity::BuffersParity()
{
  goal_init_size = 0;
  result_init_size = 0;
  index = 0;
}

void BuffersParity::init_size(uint goal_size, uint result_size)
{
  goal_init_size = goal_size;
  result_init_size = result_size;
}

void BuffersParity::create_pair()
{
  debug_log("new buffers pair created");
  uint g_ssbo = create_ssbo();
  uint r_ssbo = create_ssbo();
  result_ssbo.push_back(r_ssbo);
  goal_ssbo.push_back(g_ssbo);
  store_ssbo(r_ssbo, NULL, result_init_size, GL_DYNAMIC_DRAW);
  store_ssbo(g_ssbo, NULL, goal_init_size, GL_DYNAMIC_READ);
  occupancy.push_back(0);
}

bool BuffersParity::ready(uint idx)
{
  return occupancy[idx] == 0 || goal_ssbo.size() <= idx;
}

uint BuffersParity::get_size()
{
  return goal_ssbo.size();
}

uint BuffersParity::get_goal()
{
  return goal_ssbo[index];
}

uint BuffersParity::get_result()
{
  return result_ssbo[index];
}

void BuffersParity::store_goal(void *data, uint item_size, uint item_count)
{
  store_ssbo(goal_ssbo[index], data, item_size * item_count, GL_DYNAMIC_DRAW);
  occupancy[index] = item_count;
}

void BuffersParity::retrieve_result(void *data, uint item_size, uint &item_count)
{
  item_count = occupancy[index];
  retrieve_ssbo(result_ssbo[index], data, item_size * item_count);
  occupancy[index] = 0;
}

void BuffersParity::bind_pair(uint idx, uint goal_binding, uint result_binding)
{
  while (idx >= goal_ssbo.size())
  {
    create_pair();
  }
  index = idx;
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, result_binding, result_ssbo[index]);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, goal_binding, goal_ssbo[index]);
}

void BuffersParity::set_index(uint idx)
{
  while (idx >= goal_ssbo.size())
  {
    create_pair();
  }
  index = idx;
}

void BuffersParity::set_sync()
{
  GLsync s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (sync.size() <= index)
  {
    sync.push_back(s);
  }
  else
  {
    sync[index] = s;
  }
}

bool BuffersParity::wait_sync(uint idx, GLuint64 time_out)
{
  uint res = glClientWaitSync(sync[idx], 0, time_out);
  return res == GL_ALREADY_SIGNALED || res == GL_CONDITION_SATISFIED;
}

AnimationIndex solve_motion_matching_cs(
  AnimationDataBasePtr dataBase,
  const AnimationIndex &index,
  const AnimationGoal &goal,
  MatchingScores &best_score,
  int& charId,
  GoalsBuffer& goal_buffer,
  ResultsBuffer& result_buffer)
{
  goal_buffer.push(goal, index.get_clip_index(), index.get_cadr_index(), best_score, charId);
  if (result_buffer.ready(charId))
  {
    return result_buffer.get(charId, best_score);
  }
  return AnimationIndex(dataBase, index.get_clip_index(), index.get_cadr_index());
}