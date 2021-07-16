#pragma once
#include "Animation/AnimationTree/animation_tree.h"
#include "Engine/Render/mesh.h"
#include "Engine/Render/Shader/shader.h"
#include "Engine/Render/material.h"
#include "Engine/Render/direction_light.h"
#include "Engine/transform.h"
class AnimationRender
{
private:
  MeshPtr mesh;
  Asset<Material> material;
  Shader shader;
public:
  AnimationRender() = default;
  AnimationRender(MeshPtr mesh_ptr, Asset<Material> materail_ptr, const Shader& shader);
  void render(const Transform &transform, const mat4 view_projection, const vec3 &camera_position,  const DirectionLight& light, const AnimationTree &tree, bool wire_frame = false) const ;
  Asset<Material> get_material() const;
};
