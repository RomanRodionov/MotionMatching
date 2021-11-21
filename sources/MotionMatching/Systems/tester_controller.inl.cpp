#include "tester_controller.inl"
//Code-generator production

void tester_update_func();

ecs::SystemDescription tester_update_descr("tester_update", {
  {ecs::get_type_description<ecs::EntityId>("eid"), false},
  {ecs::get_type_description<AnimationTester>("animationTester"), false},
  {ecs::get_type_description<AnimationPlayer>("animationPlayer"), false}
}, tester_update_func, ecs::SystemOrder::LOGIC, ecs::SystemTag::Game);

void tester_update_func()
{
  ecs::perform_system(tester_update_descr, tester_update);
}

void test_count_func();

ecs::SystemDescription test_count_descr("test_count", {
  {ecs::get_type_description<vector<ecs::EntityId>>("testers"), false},
  {ecs::get_type_description<Settings>("settings"), false}
}, test_count_func, ecs::SystemOrder::NO_ORDER, ecs::SystemTag::Game);

void test_count_func()
{
  ecs::perform_system(test_count_descr, test_count);
}

void start_test_handler(const ecs::OnEntityCreated &event);
void start_test_singl_handler(const ecs::OnEntityCreated &event, ecs::EntityId eid);

ecs::EventDescription<ecs::OnEntityCreated> start_test_descr("start_test", {
  {ecs::get_type_description<AnimationTester>("animationTester"), false},
  {ecs::get_type_description<AnimationPlayer>("animationPlayer"), false},
  {ecs::get_type_description<Transform>("transform"), false},
  {ecs::get_type_description<PersonController>("personController"), false},
  {ecs::get_type_description<Settings>("settings"), false}
}, start_test_handler, start_test_singl_handler, ecs::SystemTag::GameEditor);

void start_test_handler(const ecs::OnEntityCreated &event)
{
  ecs::perform_event(event, start_test_descr, start_test);
}
void start_test_singl_handler(const ecs::OnEntityCreated &event, ecs::EntityId eid)
{
  ecs::perform_event(event, start_test_descr, eid, start_test);
}


