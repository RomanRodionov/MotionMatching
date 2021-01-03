#pragma once
#include "CommonCode/common.h"
#include "CommonCode/GameObject/game_object.h"
#include "CommonCode/Light/direction_light.h"
#include "CommonCode/Animation/animation_preprocess.h"
#include <vector>
class Scene
{
private:
  vector<GameObjectPtr> gameObjects;
  AnimationPlayerPtr animPlayer;
  DirectionLight sun;
public:
  void init();
  void update();
  void render();

};