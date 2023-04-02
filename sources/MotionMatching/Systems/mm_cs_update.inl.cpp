#include "mm_cs_update.inl"
#include <ecs_perform.h>
//Code-generator production

void init_cs_data_func();

ecs::SystemDescription init_cs_data_descr("init_cs_data", {
  {ecs::get_type_description<Asset<AnimationDataBase>>("dataBase"), false},
  {ecs::get_type_description<bool>("mm_mngr"), false},
  {ecs::get_type_description<int>("groups_per_char"), false},
  {ecs::get_type_description<int>("group_size"), false},
  {ecs::get_type_description<CSData>("cs_data"), false},
  {ecs::get_type_description<SettingsContainer>("settingsContainer"), false},
  {ecs::get_type_description<int>("mmIndex"), true}
}, {
}, {},
{},
{"motion_matching_cs_update", "motion_matching_update"},
init_cs_data_func, "act", {}, false);

void init_cs_data_func()
{
  ecs::perform_system(init_cs_data_descr, init_cs_data);
}

void motion_matching_cs_update_func();

ecs::SystemDescription motion_matching_cs_update_descr("motion_matching_cs_update", {
  {ecs::get_type_description<Asset<AnimationDataBase>>("dataBase"), false},
  {ecs::get_type_description<bool>("mm_mngr"), false},
  {ecs::get_type_description<GoalsBuffer>("goal_buffer"), false},
  {ecs::get_type_description<ResultsBuffer>("result_buffer"), false},
  {ecs::get_type_description<CSData>("cs_data"), false},
  {ecs::get_type_description<SettingsContainer>("settingsContainer"), false},
  {ecs::get_type_description<int>("mmIndex"), true}
}, {
}, {},
{"animation_player_update"},
{},
motion_matching_cs_update_func, "act", {}, false);

void motion_matching_cs_update_func()
{
  ecs::perform_system(motion_matching_cs_update_descr, motion_matching_cs_update);
}

void motion_matching_cs_retrieve_func();

ecs::SystemDescription motion_matching_cs_retrieve_descr("motion_matching_cs_retrieve", {
  {ecs::get_type_description<Asset<AnimationDataBase>>("dataBase"), false},
  {ecs::get_type_description<bool>("mm_mngr"), false},
  {ecs::get_type_description<GoalsBuffer>("goal_buffer"), false},
  {ecs::get_type_description<ResultsBuffer>("result_buffer"), false},
  {ecs::get_type_description<CSData>("cs_data"), false},
  {ecs::get_type_description<SettingsContainer>("settingsContainer"), false},
  {ecs::get_type_description<int>("mmIndex"), true}
}, {
}, {},
{},
{},
motion_matching_cs_retrieve_func, "animation", {}, false);

void motion_matching_cs_retrieve_func()
{
  ecs::perform_system(motion_matching_cs_retrieve_descr, motion_matching_cs_retrieve);
}


