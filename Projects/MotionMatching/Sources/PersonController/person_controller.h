#pragma once
#include "3dmath.h"
class AnimationPlayer;
class Transform;
class PersonController
{
public:
  float simulatedRotation, realRotation, wantedRotation, angularSpeed;
  vec3 speed, simulatedPosition, realPosition;
  bool disableEvents, crouching;
  int rotationStrafe;
  PersonController(vec3 position);
  void update_from_speed(const AnimationPlayer &player, Transform &transform, vec3 speed, float dt);
  void set_pos_rotation(Transform &transform, vec3 position, float rotation);
};
struct KeyboardEvent;
struct MouseMoveEvent;
struct ControllerKeyBoardEvent
{
  KeyboardEvent e;
  ControllerKeyBoardEvent(const KeyboardEvent &e):e(e){}
};
struct ControllerMouseMoveEvent
{
  MouseMoveEvent e;
  ControllerMouseMoveEvent(const MouseMoveEvent &e):e(e){}
};