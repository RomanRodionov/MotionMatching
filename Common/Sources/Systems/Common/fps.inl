#include "ecs/ecs.h"
#include "Engine/time.h"
#include "Engine/imgui/imgui.h"

SYSTEM(ecs::SystemOrder::UI) fps_ui(float &fps)
{
  ImGui::Begin("fps");
  ImGui::Text("%.1f", fps = Time::fps());
  ImGui::End();
}