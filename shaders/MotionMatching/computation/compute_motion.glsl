#shader compute_motion

#compute_shader

#define INF 1e10
#define GROUP_SIZE 256

const uint nodesCount = 4;
const uint pathLength = 3;

struct Tag
{
    uint tags[2];
};

struct FeatureCell
{
  vec4 nodes[nodesCount];
  vec4 nodesVelocity[nodesCount];
  vec4 points[pathLength];
  vec4 pointsVelocity[pathLength];
  vec4 angularVelocity;
  vec4 weights;
  //float goalPathMatchingWeight;
  //float realism;
  //uint padding[3];
};

struct GoalBuffer
{
  vec4 nodes[nodesCount];
  vec4 nodesVelocity[nodesCount];
  vec4 points[pathLength];
  vec4 pointsVelocity[pathLength];
  vec4 angularVelocity;
  //uint padding[1];
}; 
 /*
  Tag tag;
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  */

struct MatchingScores
{
  float pose, goal_tag, goal_path, trajectory_v, trajectory_w;
  float full_score;
  uint idx;
  uint padding;
};

layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
/*
layout(std430, binding = 0) buffer inout_values
{
    float data_SSBO[];
};
*/
layout(std430, binding = 0) buffer mm_data
{
    FeatureCell feature[];
};
layout(std430, binding = 1) buffer result_data
{
    MatchingScores results[];
};
layout (std140, binding = 2) uniform DataBlock
{
    GoalBuffer goal_data;
};  

uniform int data_size;
uniform int iterations;

/*
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
*/

float pose_matching_norma(in FeatureCell feature, in GoalBuffer goal)
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

float goal_path_norma(in FeatureCell feature, in GoalBuffer goal)
{
  float path_norma = 0.f;
  float distScale = length(goal.points[pathLength - 1] + feature.points[pathLength - 1]) * 0.5f;
  for (uint i = 0; i < pathLength; i++)
    path_norma += length(goal.points[i] - feature.points[i]);
  return path_norma / (0.1f + distScale);
}

float trajectory_v_norma(in FeatureCell feature, in GoalBuffer goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < pathLength; i++)
    path_norma += length(goal.pointsVelocity[i] - feature.pointsVelocity[i]);
  return path_norma;
}

float trajectory_w_norma(in FeatureCell feature, in GoalBuffer goal)
{
  float path_norma = 0.f;
  for (uint i = 0; i < pathLength; i++)
    path_norma += abs(goal.angularVelocity[i] - feature.angularVelocity[i]);
  return path_norma;
}

MatchingScores get_score(in FeatureCell feature, in GoalBuffer goal)
{
  MatchingScores score;
  score.pose = pose_matching_norma(feature, goal);
  score.goal_path = goal_path_norma(feature, goal) * feature.weights[0];
  score.trajectory_v = trajectory_v_norma(feature, goal);
  score.trajectory_w = trajectory_w_norma(feature, goal);
  score.full_score = score.pose * feature.weights[1] + score.goal_path + score.trajectory_v + score.trajectory_w;
  return score;
}

shared MatchingScores min_scores[GROUP_SIZE];

void main()
{
  MatchingScores score;
  min_scores[gl_LocalInvocationID.x].full_score = INF;
  for (uint i = 0; (i < iterations) && (gl_GlobalInvocationID.x * iterations + i < data_size); i++)
  {
    score = get_score(feature[gl_GlobalInvocationID.x * iterations + i], goal_data);
    if (i == 0 || min_scores[gl_LocalInvocationID.x].full_score > score.full_score)
    {
      score.idx = gl_GlobalInvocationID.x * iterations + i;
      min_scores[gl_LocalInvocationID.x] = score;
    }
  }
  uint step = GROUP_SIZE / 2;
  uint arr_size = data_size / iterations;
  if (arr_size % iterations > 0)
    arr_size++;
  memoryBarrierShared();
  barrier();
  while (step > 0) {
    if ((gl_LocalInvocationID.x < step) && (gl_GlobalInvocationID.x + step < arr_size)) 
    {
      if (min_scores[gl_LocalInvocationID.x + step].full_score < min_scores[gl_LocalInvocationID.x].full_score) 
        min_scores[gl_LocalInvocationID.x] = min_scores[gl_LocalInvocationID.x + step];
    }
    step /= 2;
    memoryBarrierShared();
    barrier();
  }
  if (gl_LocalInvocationID.x == 0)
    results[gl_WorkGroupID.x] = min_scores[0];
  //results[gl_WorkGroupID.x].full_score = goal_data.angularVelocity.x;
  memoryBarrierShared();
  barrier();
}
