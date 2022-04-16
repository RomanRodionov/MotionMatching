#include "third_person_controller.inl"
//Code-generator production

ecs::QueryDescription update_attached_camera_descr("update_attached_camera", {
  {ecs::get_type_description<Transform>("transform"), false}
}, {
});

template<typename Callable>
void update_attached_camera(ecs::EntityId eid, Callable lambda)
{
  ecs::perform_query<Transform&>
  (update_attached_camera_descr, eid, lambda);
}


void third_peson_controller_update_func();

ecs::SystemDescription third_peson_controller_update_descr("third_peson_controller_update", {
  {ecs::get_type_description<ecs::EntityId>("attachedCamera"), false},
  {ecs::get_type_description<PersonController>("personController"), false},
  {ecs::get_type_description<ThirdPersonController>("thirdPersonController"), false}
}, {
}, {},
third_peson_controller_update_func, ecs::SystemOrder::LOGIC, ecs::SystemTag::Game,
{},
{});

void third_peson_controller_update_func()
{
  ecs::perform_system(third_peson_controller_update_descr, third_peson_controller_update);
}

void third_controller_appear_handler(const ecs::OnEntityCreated &event);
void third_controller_appear_singl_handler(const ecs::OnEntityCreated &event, ecs::EntityId eid);

ecs::EventDescription<ecs::OnEntityCreated> third_controller_appear_descr("third_controller_appear", {
  {ecs::get_type_description<ecs::EntityId>("attachedCamera"), false},
  {ecs::get_type_description<ThirdPersonController>("thirdPersonController"), false}
}, {
}, {},
third_controller_appear_handler, third_controller_appear_singl_handler, ecs::SystemTag::Game);

void third_controller_appear_handler(const ecs::OnEntityCreated &event)
{
  ecs::perform_event(event, third_controller_appear_descr, third_controller_appear);
}
void third_controller_appear_singl_handler(const ecs::OnEntityCreated &event, ecs::EntityId eid)
{
  ecs::perform_event(event, third_controller_appear_descr, eid, third_controller_appear);
}

void mouse_move_handler_handler(const MouseMoveEvent &event);
void mouse_move_handler_singl_handler(const MouseMoveEvent &event, ecs::EntityId eid);

ecs::EventDescription<MouseMoveEvent> mouse_move_handler_descr("mouse_move_handler", {
  {ecs::get_type_description<ecs::EntityId>("eid"), false},
  {ecs::get_type_description<ThirdPersonController>("thirdPersonController"), false},
  {ecs::get_type_description<PersonController>("personController"), false},
  {ecs::get_type_description<Settings>("settings"), false}
}, {
}, {},
mouse_move_handler_handler, mouse_move_handler_singl_handler, ecs::SystemTag::Game);

void mouse_move_handler_handler(const MouseMoveEvent &event)
{
  ecs::perform_event(event, mouse_move_handler_descr, mouse_move_handler);
}
void mouse_move_handler_singl_handler(const MouseMoveEvent &event, ecs::EntityId eid)
{
  ecs::perform_event(event, mouse_move_handler_descr, eid, mouse_move_handler);
}

void mouse_wheel_handler_handler(const MouseWheelEvent &event);
void mouse_wheel_handler_singl_handler(const MouseWheelEvent &event, ecs::EntityId eid);

ecs::EventDescription<MouseWheelEvent> mouse_wheel_handler_descr("mouse_wheel_handler", {
  {ecs::get_type_description<ThirdPersonController>("thirdPersonController"), false}
}, {
}, {},
mouse_wheel_handler_handler, mouse_wheel_handler_singl_handler, ecs::SystemTag::Game);

void mouse_wheel_handler_handler(const MouseWheelEvent &event)
{
  ecs::perform_event(event, mouse_wheel_handler_descr, mouse_wheel_handler);
}
void mouse_wheel_handler_singl_handler(const MouseWheelEvent &event, ecs::EntityId eid)
{
  ecs::perform_event(event, mouse_wheel_handler_descr, eid, mouse_wheel_handler);
}

void crouch_event_handler_handler(const KeyEventAnyActionKey &event);
void crouch_event_handler_singl_handler(const KeyEventAnyActionKey &event, ecs::EntityId eid);

ecs::EventDescription<KeyEventAnyActionKey> crouch_event_handler_descr("crouch_event_handler", {
  {ecs::get_type_description<ecs::EntityId>("eid"), false},
  {ecs::get_type_description<ThirdPersonController>("thirdPersonController"), false}
}, {
}, {},
crouch_event_handler_handler, crouch_event_handler_singl_handler, ecs::SystemTag::Game);

void crouch_event_handler_handler(const KeyEventAnyActionKey &event)
{
  ecs::perform_event(event, crouch_event_handler_descr, crouch_event_handler);
}
void crouch_event_handler_singl_handler(const KeyEventAnyActionKey &event, ecs::EntityId eid)
{
  ecs::perform_event(event, crouch_event_handler_descr, eid, crouch_event_handler);
}

void animation_player_handler_handler(const KeyEventAnyActionKey &event);
void animation_player_handler_singl_handler(const KeyEventAnyActionKey &event, ecs::EntityId eid);

ecs::EventDescription<KeyEventAnyActionKey> animation_player_handler_descr("animation_player_handler", {
  {ecs::get_type_description<AnimationPlayer>("animationPlayer"), false},
  {ecs::get_type_description<ThirdPersonController>("thirdPersonController"), false}
}, {
}, {},
animation_player_handler_handler, animation_player_handler_singl_handler, ecs::SystemTag::Game);

void animation_player_handler_handler(const KeyEventAnyActionKey &event)
{
  ecs::perform_event(event, animation_player_handler_descr, animation_player_handler);
}
void animation_player_handler_singl_handler(const KeyEventAnyActionKey &event, ecs::EntityId eid)
{
  ecs::perform_event(event, animation_player_handler_descr, eid, animation_player_handler);
}


