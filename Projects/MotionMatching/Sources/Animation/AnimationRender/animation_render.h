#pragma once
#include "Animation/AnimationTree/animation_tree.h"
#include <Engine/Render/mesh.h>
#include <Engine/Render/Shader/shader.h>
#include <Engine/Render/material.h>
#include <Engine/transform.h>
#include <Serialization/reflection.h>
class AnimationRender
{
private:
  REFLECT(MeshRender,
  (Asset<Mesh>) (mesh),
  (Asset<Material>) (material)
  )

public:
  AnimationRender() = default;
  AnimationRender(Asset<Mesh> mesh, Asset<Material> material);
  void render(const Transform &transform, const AnimationTree &tree, bool wire_frame = false) const ;
  const Asset<Material>& get_material() const;
  Asset<Material>& get_material();
};
