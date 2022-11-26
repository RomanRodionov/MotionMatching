#version 450
#define CS 1




const uint nodesCount = 4;
const uint pathLength = 3;

struct Tag
{
    uint tags[2];
};

struct FeatureCell
{
  vec3 nodes[nodesCount];
  vec3 nodesVelocity[nodesCount];
  vec3 points[pathLength];
  vec3 pointsVelocity[pathLength];
  float angularVelocity[pathLength];
  float goalPathMatchingWeight;
  Tag tag;
  //uint padding[];
};

struct InOutBuffer
{
  vec3 nodes[nodesCount];
  vec3 nodesVelocity[nodesCount];
  vec3 points[pathLength];
  vec3 pointsVelocity[pathLength];
  float angularVelocity[pathLength];
  Tag tag;
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint padding[3];
};

struct BufferMatchingScores
{
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint padding[2];
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer inout_values
{
    float data_SSBO[];
};
layout(std430, binding = 1) buffer mm_data
{
    FeatureCell feature[];
};
layout(std430, binding = 2) buffer goal_data
{
    InOutBuffer goal;
};
layout(std430, binding = 3) buffer result_data
{
    BufferMatchingScores results[];
};


uniform uint arr_size;
uniform uint iterations;

struct MatchingScores
{
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
};

struct ArgMin
{
  float value;
  uint clip, frame;
  MatchingScores score;
};

float pose_matching_norma(in FeatureCell feature, in InOutBuffer goal)
{
  float pose_norma = 0.f, vel_norma = 0.f;
  for (int i = 0; i < nodesCount; i++)
  {
    pose_norma += length(feature.nodes[i] - goal.nodes[i]);
    vel_norma += length(feature.nodesVelocity[i] - goal.nodesVelocity[i]);
  }
  return pose_norma + vel_norma;
}

bool has_goal_tags(in Tag tag1, in Tag tag2)
{
  return tag1.tags[0] == tag2.tags[0] && tag1.tags[1] == tag2.tags[1];
}

float goal_path_norma(in FeatureCell feature, in InOutBuffer goal)
{
  float path_norma = 0.f;
  float distScale = length(goal.points[pathLength - 1] + feature.points[pathLength - 1]) * 0.5f;
  for (uint i = 0; i < pathLength; i++)
    path_norma += length(goal.points[i] - feature.points[i]);
  return path_norma / (0.1f + distScale);
}

float trajectory_v_norma(in FeatureCell feature, in InOutBuffer goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < pathLength; i++)
    path_norma += length(goal.pointsVelocity[i] - feature.pointsVelocity[i]);
  return path_norma;
}

float trajectory_w_norma(in FeatureCell feature, in InOutBuffer goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < pathLength; i++)
    path_norma += length(goal.angularVelocity[i] - feature.angularVelocity[i]);
  return path_norma;
}

BufferMatchingScores get_score(in FeatureCell feature, in InOutBuffer goal)
{
  BufferMatchingScores score;
  score.pose = pose_matching_norma(feature, goal);
  score.goal_path = goal_path_norma(feature, goal) * feature.goalPathMatchingWeight;
  score.trajectory_v = 1;//trajectory_v_norma(feature, goal);
  score.trajectory_w = trajectory_w_norma(feature, goal);
  score.full_score = score.pose + score.goal_path + score.trajectory_v + score.trajectory_w;
  return score;
}

void main()
{
  for (uint i = 0; i < iterations && gl_GlobalInvocationID.x * iterations + i < arr_size; i++)
  {
    results[gl_GlobalInvocationID.x * iterations + i] = get_score(feature[gl_GlobalInvocationID.x * iterations + i], goal);
  }
  results[gl_GlobalInvocationID].trajectory_v = 1;
  memoryBarrierShared();
  barrier();
}

/*
shared float values[512];

void main()
{
  uint step = 512;
  uint left_border = (gl_GlobalInvocationID.x - gl_LocalInvocationID.x) * 2;
  uint base_idx = left_border + gl_LocalInvocationID.x;
  if (base_idx < arr_size) 
  {
    values[gl_LocalInvocationID.x] = data_SSBO[base_idx];
  }
  memoryBarrierShared();
  barrier();
  if (base_idx + step < arr_size) 
  {
    float value = data_SSBO[base_idx + step];
    if (value < values[gl_LocalInvocationID.x]) values[gl_LocalInvocationID.x] = value;
  }
  memoryBarrierShared();
  barrier();
  while (step > 0) {
    if ((gl_LocalInvocationID.x < step) && (base_idx + step < arr_size)) 
    {
      float val1 = values[gl_LocalInvocationID.x];
      float val2 = values[gl_LocalInvocationID.x + step];
      if (val2 < val1) values[gl_LocalInvocationID.x] = val2;
    }
      step /= 2;
      memoryBarrierShared();
      barrier();
  }
  data_SSBO[base_idx] = values[gl_LocalInvocationID.x];
}
*/