#pragma once
#include "Mesh/mesh.h"
#include "Shader/shader.h"
#include "Material/material.h"
#include "Light/direction_light.h"
#include "Transform/transform.h"
class MeshRender
{
private:
  MeshPtr mesh;
  MaterialPtr material;
  Shader shader;
public:
 MeshRender(MeshPtr mesh_ptr, MaterialPtr materail_ptr, const Shader& shader);
  void render(const Transform &transform, const mat4 view_projection, const vec3 &camera_position, const DirectionLight& light, bool wire_frame = false) const;
  MaterialPtr get_material() const;
  const Shader& get_shader() const;
  Shader& get_shader();
};
MeshRender create_plane(bool create_uv = false);
MeshRender create_cube(bool create_uv = false);
MeshRender create_sphere(int detailed = 1, bool smooth = false, bool create_uv = false);
MeshRender create_postfx();
