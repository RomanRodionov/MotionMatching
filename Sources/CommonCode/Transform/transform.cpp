#include "transform.h"

Transform::Transform(vec3 position, mat4x4 rotation, vec3 scale):
  position(position), rotation(rotation), scale(scale)
{}
Transform::Transform(vec3 position, vec3 rotation, vec3 scale):
  position(position), rotation(glm::yawPitchRoll(rotation.x, rotation.y, rotation.z)), scale(scale)
{}

Transform::Transform():
position(vec3()), rotation(mat4x4(1.f)), scale(vec3(1.f))
{}
vec3& Transform::get_position()
{
  return position;
}
const vec3& Transform::get_position() const 
{
  return position;
}
mat4x4 Transform::get_transform() const
{
  mat4 s = glm::scale(mat4(1.f), scale);
  mat4 t = glm::translate(mat4(1.f), position);
  return t * s * rotation;
}
vec3 Transform::get_forward() const
{
  return -rotation[2];
  //return vec3(rotation[0][2], rotation[1][2],rotation[2][2]);
}
vec3 Transform::get_right() const
{
  return rotation[0];
  //return -vec3(rotation[0][0], rotation[1][0],rotation[2][0]);
}
vec3 Transform::get_up() const
{
  return rotation[1];
  //return -vec3(rotation[0][1], rotation[1][1],rotation[2][1]);
}
void Transform::set_rotation(float yaw, float pitch, float roll)
{
  rotation = glm::yawPitchRoll(yaw, pitch, roll);
}
void Transform::set_rotation(const mat4& rotation)
{
  this->rotation = rotation;
}
void Transform::set_position(const vec3 position)
{
  this->position = position;
}
void Transform::set_scale(const vec3 scale)
{
  this->scale = scale;
}

void Transform::set_to_shader(const Shader& shader)
{
  shader.set_mat4x4("Model", get_transform());
}