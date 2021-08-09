#include "animation_render.h"
#include <Engine/camera.h>

static vector<mat4> curTransform;
AnimationRender::AnimationRender(Asset<Mesh> mesh, Asset<Material> material):
    mesh(mesh), material(material)
  {}

void AnimationRender::render(const Transform &transform, const AnimationTree &tree, bool wire_frame) const
{
  if (!mesh)
    return;
  curTransform.resize(mesh->bonesMap.size());
  for (uint i = 0; i < tree.nodes.size(); i++)
  {
    auto it2 = mesh->bonesMap.find(tree.nodes[i].get_name());
    if (it2 != mesh->bonesMap.end())
    {
      curTransform[it2->second] = tree.get_bone_transform(i);
    }
  }
  const Shader &shader = material ? material->get_shader() : Shader();
  shader.use();
  shader.set_mat4x4("Bones", curTransform, false);
  if (material)
    material->bind_to_shader();
  transform.set_to_shader(shader);

  mesh->render(wire_frame);
  if (material)
    material->unbind_to_shader();
}
const Asset<Material>& AnimationRender::get_material() const
{
  return material;
}
Asset<Material>& AnimationRender::get_material()
{
  return material;
}