#include "motion_matching.inl"
#include <ecs_perform.h>
//Code-generator production

void motion_matching_cs_update_func();

ecs::SystemDescription motion_matching_cs_update_descr("motion_matching_cs_update", {
  {ecs::get_type_description<GoalsBuffer>("goal_buffer"), false},
  {ecs::get_type_description<ResultsBuffer>("result_buffer"), false}
}, {
}, {},
{"animation_player_update"},
{"motion_matching_update"},
motion_matching_cs_update_func, "act", {}, false);

void motion_matching_cs_update_func()
{
  ecs::perform_system(motion_matching_cs_update_descr, motion_matching_cs_update);
}

void motion_matching_update_func();

ecs::SystemDescription motion_matching_update_descr("motion_matching_update", {
  {ecs::get_type_description<Transform>("transform"), false},
  {ecs::get_type_description<AnimationPlayer>("animationPlayer"), false},
  {ecs::get_type_description<Asset<Material>>("material"), false},
  {ecs::get_type_description<int>("mmIndex"), true},
  {ecs::get_type_description<int>("mmOptimisationIndex"), false},
  {ecs::get_type_description<bool>("updateMMStatistic"), false},
  {ecs::get_type_description<Settings>("settings"), false},
  {ecs::get_type_description<SettingsContainer>("settingsContainer"), false},
  {ecs::get_type_description<MMProfiler>("profiler"), false},
  {ecs::get_type_description<MainCamera>("mainCamera"), false},
  {ecs::get_type_description<GoalsBuffer>("goal_buffer"), false},
  {ecs::get_type_description<ResultsBuffer>("result_buffer"), false},
  {ecs::get_type_description<ecs::EntityId>("eid"), false}
}, {
}, {},
{"animation_player_update"},
{},
motion_matching_update_func, "act", {}, false);

void motion_matching_update_func()
{
  ecs::perform_system(motion_matching_update_descr, motion_matching_update);
}

void init_cs_data_handler(const ecs::Event &event);
void init_cs_data_singl_handler(const ecs::Event &event, ecs::EntityId eid);

ecs::EventDescription init_cs_data_descr(
  ecs::get_mutable_event_handlers<ecs::OnEntityCreated>(), "init_cs_data", {
  {ecs::get_type_description<Asset<AnimationDataBase>>("dataBasePtr"), false},
  {ecs::get_type_description<bool>("mm_mngr"), false},
  {ecs::get_type_description<CSData>("cs_data"), false},
  {ecs::get_type_description<SettingsContainer>("settingsContainer"), false},
  {ecs::get_type_description<int>("mmIndex"), true},
  {ecs::get_type_description<int>("mmOptimisationIndex"), false}
}, {
}, {},
{},
{},
init_cs_data_handler, init_cs_data_singl_handler, {});

void init_cs_data_handler(const ecs::Event &event)
{
  ecs::perform_event((const ecs::OnEntityCreated&)event, init_cs_data_descr, init_cs_data);
}
void init_cs_data_singl_handler(const ecs::Event &event, ecs::EntityId eid)
{
  ecs::perform_event((const ecs::OnEntityCreated&)event, init_cs_data_descr, eid, init_cs_data);
}


