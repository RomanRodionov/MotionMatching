#include "ecs/ecs.h"
#include "Engine/Render/global_uniform.h"
#include "Engine/Render/render.h"

EVENT(ecs::SystemTag::GameEditor) add_global_uniform(const ecs::OnSceneCreated &)
{
  add_uniform_buffer<GlobalRenderData>("GlobalRenderData", 0);
  add_storage_buffer("InstanceData", 3952 * 110, 1);
}