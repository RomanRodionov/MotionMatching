#include "ecs/ecs.h"
#include "motion_matching_scene.h"
#include "Animation/AnimationRender/animation_render.h"
#include "Animation/AnimationRender/bone_render.h"
#include "Animation/animation_player.h"
#include "Engine/Render/mesh_render.h"
#include "Engine/camera.h"
#include "Engine/Render/debug_arrow.h"
#include "Engine/Render/skybox.h"
template<typename Callable>
void render_animation(Callable);

template<typename Callable>
void bone_render_animation(const ecs::EntityId&, Callable);

template<typename Callable>
void render_meshes(Callable);

template<typename Callable>
void render_arrows(Callable);

template<typename Callable>
void render_skybox(Callable);

template<typename Callable>
void render_debug_goal(Callable);
template<typename Callable>
void render_debug_goal_on_animplayer(Callable);


SYSTEM(ecs::SystemOrder::MIDDLE_RENDER) main_render(const SceneRender &sceneRender)
{
  mat4 camTransform, camProjection;
  if(!main_camera(camTransform, camProjection))
  {
    debug_error("Need main camera");
    return;
  }

  vec3 cameraPosition = camTransform[3];
  mat4 viewTrasform = inverse(camTransform);

  mat4 viewProjection = camProjection *  viewTrasform;
  mat4 viewProjectionSkybox = camProjection *  mat4(mat3(viewTrasform));

  const DirectionLight &light = sceneRender.sun;
  bool wire_frame = false; 
  QUERY() render_animation([&](
    ecs::EntityId eid,
    const AnimationRender &animationRender,
    const AnimationPlayer &animationPlayer,
    const Transform &transform)
  {
    animationRender.render(transform, viewProjection, cameraPosition, light, animationPlayer.get_tree(), wire_frame);

    QUERY() bone_render_animation(eid, [&](
      const BoneRender &boneRender)
    {
      boneRender.render(transform.get_transform(), animationPlayer.get_tree());

    });
  });

  QUERY() render_meshes([&](
    const MeshRender &meshRender,
    const Transform &transform)
  {
    meshRender.render(transform, viewProjection, cameraPosition, light, wire_frame);
  });



  QUERY() render_debug_goal([&](Transform &debugTransform, const MeshRender &debugGoalSphere)
  {
    QUERY() render_debug_goal_on_animplayer([&](
      const AnimationPlayer &animationPlayer,
      const Transform &transform)
    {
      AnimationLerpedIndex index = animationPlayer.get_motion_matching() ? animationPlayer.get_motion_matching()->get_index() : animationPlayer.get_index();

      mat4 transformation = transform.get_transform();


      MaterialPtr material = debugGoalSphere.get_material();
      const auto& feature = index.first.get_feature();
      AnimationTrajectory trajectory = index.first.get_trajectory();
      
      /*vec3 man = transformation * vec4(feature.features[(int)AnimationFeaturesNode::Hips], 1.f);
      Ray ray(man, vec3(0,-1,0), 100);
      Collision collision = ray_cast(ray);
      draw_arrow(ray.from, collision.collisionPoint, vec3(10,0,0), 0.04f, false);*/

      u8 onGround = index.first.get_clip().onGround[index.first.get_cadr_index()];
      onGround = animationPlayer.onGround;
      #define DEBUG_NODE(node)\
      {\
        debugTransform.get_position() = transformation * vec4(feature.features[(int)node], 1.f);\
        debugGoalSphere.render(debugTransform, viewProjection, cameraPosition, light, true);\
      }
      #define DEBUG_NODE_SPEED(node, speed)\
      {\
        vec3 p = vec3(transformation * vec4(feature.features[(int)node], 1.f));\
        vec3 v = vec3(transformation * vec4(feature.features[(int)speed], 0.f));\
        draw_arrow(p, p + v * 5.f, vec3(1,0,0), 0.02f, false);\
      }

      material->set_property(Property("Ambient", vec3(1,1,1)));
      debugTransform.set_scale(vec3(0.1f));
      DEBUG_NODE(AnimationFeaturesNode::LeftToeBase)
      DEBUG_NODE(AnimationFeaturesNode::LeftHand)
      DEBUG_NODE(AnimationFeaturesNode::RightToeBase)
      DEBUG_NODE(AnimationFeaturesNode::RightHand)
      DEBUG_NODE(AnimationFeaturesNode::Hips)
      
      DEBUG_NODE_SPEED(AnimationFeaturesNode::LeftToeBase, AnimationFeaturesNode::LeftToeSpeed)
      DEBUG_NODE_SPEED(AnimationFeaturesNode::RightToeBase, AnimationFeaturesNode::RightToeSpeed)

      if (onGround & 1)
      {
        material->set_property(Property("Ambient", vec3(1,0,0)));
        debugTransform.set_scale(vec3(0.11f));
        
        debugTransform.get_position() = transformation * vec4(feature.features[(int)AnimationFeaturesNode::LeftToeBase], 1.f);
        debugGoalSphere.render(debugTransform, viewProjection, cameraPosition, light, true);
      }
      if (onGround & 2)
      {
        material->set_property(Property("Ambient", vec3(1,0,0)));
        debugTransform.set_scale(vec3(0.11f));
        
        debugTransform.get_position() = transformation * vec4(feature.features[(int)AnimationFeaturesNode::RightToeBase], 1.f);
        debugGoalSphere.render(debugTransform, viewProjection, cameraPosition, light, true);
      }
      constexpr float dirLength = 0.3f;
      constexpr vec3 colors[2] = {vec3(0,1,0), vec3(1,0,0)};
      constexpr float lenghts[2] = {0.3f, 0.3f};

      const std::array<TrajectoryPoint,AnimationTrajectory::PathLength> *trajectories[2] = {&trajectory.trajectory, &animationPlayer.inputGoal.path.trajectory};
      for(int i = 0; i < 2; i++)
      {
        for (const TrajectoryPoint &p: *trajectories[i])
        {
          vec3 v = vec3(transformation * vec4(p.point, 1.f));
          vec3 w = vec3(transformation * vec4(quat(vec3(0, p.rotation, 0)) * vec3(0, 0, dirLength * lenghts[i]), 0.f));
          draw_arrow(v, v + w, colors[i], 0.02f, false);
        }
      }
    });
  });
  QUERY() render_arrows([&](
    DebugArrow &debugArrows)
  {
    debugArrows.render(viewProjection, cameraPosition, light, wire_frame);
  });



  QUERY() render_skybox([&](
    SkyBox &skyBox)
  {
    skyBox.render(viewProjectionSkybox, cameraPosition, wire_frame);
  });
}