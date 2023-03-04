#shader compute_motion

#compute_shader

#define INF 1e16
#define GROUP_SIZE 64

#define NODES_COUNT 4
#define PATH_LENGTH 3

struct Tag
{
    uint tag1;
    uint tag2;
};

struct FeatureCell
{
  vec4 nodes[NODES_COUNT];
  vec4 nodesVelocity[NODES_COUNT];
  vec4 points[PATH_LENGTH];
  vec4 pointsVelocity[PATH_LENGTH];
  vec4 angularVelocity;
  Tag tags;
  uint padding1;
  uint padding2;
};

struct MatchingScores
{
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint idx;
  uint padding;
};

layout(local_size_x = GROUP_SIZE) in;

layout(std430, binding = 0) buffer mm_data
{
    FeatureCell feature[];
};
layout(std430, binding = 1) buffer result_data
{
    MatchingScores results[];
};
layout (std430, binding = 2) buffer DataBlock
{
    FeatureCell goal_data[];
};  

uniform int data_size;
uniform int iterations;
uniform float goalPathMatchingWeight;
uniform float realism;
uniform int queue_size;

float pose_matching_norma(in FeatureCell feature, in FeatureCell goal)
{
  float pose_norma = 0.f, vel_norma = 0.f;
  for (int i = 0; i < NODES_COUNT; i++)
  {
    pose_norma += length(feature.nodes[i] - goal.nodes[i]);
    vel_norma += length(feature.nodesVelocity[i] - goal.nodesVelocity[i]);
  }
  return pose_norma + vel_norma;
}

bool has_goal_tags(in Tag tag1, in Tag tag2)
{
  return tag1.tag1 == tag2.tag1 && tag1.tag2 == tag2.tag2;
}

float goal_path_norma(in FeatureCell feature, in FeatureCell goal)
{
  float path_norma = 0.f;
  float distScale = length(goal.points[PATH_LENGTH - 1] + feature.points[PATH_LENGTH - 1]) * 0.5f;
  for (uint i = 0; i < PATH_LENGTH; i++)
    path_norma += length(goal.points[i] - feature.points[i]);
  return path_norma / (0.1f + distScale);
}

float trajectory_v_norma(in FeatureCell feature, in FeatureCell goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < PATH_LENGTH; i++)
    path_norma += length(goal.pointsVelocity[i] - feature.pointsVelocity[i]);
  return path_norma;
}

float trajectory_w_norma(in FeatureCell feature, in FeatureCell goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < PATH_LENGTH; i++)
    path_norma += abs(goal.angularVelocity[i] - feature.angularVelocity[i]);
  return path_norma;
}

MatchingScores get_score(in FeatureCell feature, in FeatureCell goal)
{
  MatchingScores score;
  score.pose = pose_matching_norma(feature, goal);
  score.goal_path = goal_path_norma(feature, goal) * goalPathMatchingWeight;
  score.trajectory_v = trajectory_v_norma(feature, goal);
  score.trajectory_w = trajectory_w_norma(feature, goal);
  score.full_score = score.pose * realism + score.goal_path + score.trajectory_v + score.trajectory_w;
  return score;
}

shared MatchingScores min_scores[GROUP_SIZE];

void main()
{
  uint queue_index = gl_WorkGroupID.y;
  uint circle_limit;
  uint base_idx = gl_GlobalInvocationID.x * iterations;
  if (data_size > base_idx)
  {
    circle_limit = data_size - base_idx < iterations ? data_size - base_idx : iterations;
  }
  while (queue_index < queue_size)
  {
    MatchingScores score;
    min_scores[gl_LocalInvocationID.x].full_score = INF;
    FeatureCell cur_goal = goal_data[queue_index];
    for (uint i = 0; i < circle_limit; i++)
    {
      if (has_goal_tags(cur_goal.tags, feature[base_idx + i].tags))
      {
        score = get_score(feature[base_idx + i], cur_goal);
        if (min_scores[gl_LocalInvocationID.x].full_score > score.full_score)
        {
          score.idx = base_idx + i;
          min_scores[gl_LocalInvocationID.x] = score;
        }
      }
    }
    uint step = GROUP_SIZE / 2;
    uint arr_size = data_size / iterations;
    if (arr_size % iterations > 0)
      arr_size++;
    memoryBarrierShared();
    barrier();
   
    while (step > 0) {
      if ((gl_LocalInvocationID.x < step) && (gl_LocalInvocationID.x + step < arr_size)) 
        if (min_scores[gl_LocalInvocationID.x + step].full_score < min_scores[gl_LocalInvocationID.x].full_score) 
          min_scores[gl_LocalInvocationID.x] = min_scores[gl_LocalInvocationID.x + step];
      step /= 2;
      memoryBarrierShared();
      barrier();
    }
    if (gl_LocalInvocationID.x == 0)
      results[queue_index * gl_NumWorkGroups.x + gl_WorkGroupID.x] = min_scores[0];
    queue_index += gl_NumWorkGroups.y;
  }
}