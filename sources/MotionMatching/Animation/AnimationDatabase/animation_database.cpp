#include "animation_database.h"
#include <resources/resource_registration.h>
#include <imgui.h>
#include <ecs/component_editor.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <application/time.h>

int AnimationDataBase::cadr_count() const
{
  int count = 0;
  for (const AnimationClip & anim: clips)
    count += anim.duration;
  return count;
}
void reload_tree(AnimationTreeData &tree, const filesystem::path &path)
{
  string fbxPath = path.parent_path().string() + "/" + path.stem().string();
  Assimp::Importer importer;
  importer.SetPropertyBool(AI_CONFIG_IMPORT_FBX_PRESERVE_PIVOTS, false);
  importer.SetPropertyFloat(AI_CONFIG_GLOBAL_SCALE_FACTOR_KEY, 1.f);
  
  importer.ReadFile(fbxPath, aiPostProcessSteps::aiProcess_Triangulate | aiPostProcessSteps::aiProcess_LimitBoneWeights |
    aiPostProcessSteps::aiProcess_GenNormals | aiProcess_GlobalScale | aiProcess_FlipWindingOrder);

  const aiScene* scene = importer.GetScene();
  if (scene && scene->mRootNode && scene->mRootNode->mChildren[0])
    tree = AnimationTreeData(scene->mRootNode->mChildren[0]);
}

void animation_preprocess(AnimationDataBase &data_base);

void AnimationDataBase::load(const filesystem::path &, bool reload, AssetStatus &status)
{
  if (!reload)
  {
    reload_tree(tree, treeSource.asset_path());
  }
  if (needForceReload)
    animation_preprocess(*this);
  register_tags(tagsNames);

  if (matchingScore.empty() || needForceReload)
  {
    matchingScore.resize(clips.size());
    for (uint i = 0; i < matchingScore.size(); i++)
      matchingScore[i].resize(clips[i].duration, 0);
  }

  status = AssetStatus::Loaded;
}
bool AnimationDataBase::edit()
{
  bool changeTree = edit_component(treeSource, "tree", false);
  if (changeTree)
    reload_tree(tree, treeSource.asset_path());
  needForceReload = ImGui::Button("force reload");

  for (auto &clip : clips)
  {
    ImGui::Text("%s tags = %s, duration = %d",
    clip.name.c_str(), clip.tags.to_string().c_str(), clip.duration);
  }

  return changeTree | needForceReload;
}; 

void AnimationDataBase::apply_settings(const MotionMatchingSettings &mmsettings, bool check_state)
{
  if (check_state)
  {
    if (!modifiedClips.empty())
      return;
  }
  if (!mmsettings.applySettingsOnce)
  {
    return;
  }
  float poseWeight = mmsettings.poseMatchingWeight;
  float velocityWeight = mmsettings.velocityMatchingWeight;
  for (uint nextClip = 0; nextClip < clips.size(); nextClip++)
  {
    AnimationClip clip = clips[nextClip];
    for (uint nextCadr = 0, n = clip.duration; nextCadr < n; nextCadr++)
    {
      auto &frame = clip.features[nextCadr];
      for (uint node = 0; node < (uint)AnimationFeaturesNode::Count; node++)
      {
        frame.features.nodes[node] = frame.features.nodes[node] * float(mmsettings.nodeWeights[node]) * poseWeight;
        if (mmsettings.velocityMatching)
          frame.features.nodesVelocity[node] = frame.features.nodesVelocity[node] * float(mmsettings.velocitiesWeights[node]) * velocityWeight;
        else
          frame.features.nodesVelocity[node] = vec3(0, 0, 0);
      }
      for (uint point = 0; point < (uint)AnimationTrajectory::PathLength; point++)
      {
        frame.trajectory.trajectory[point].velocity = frame.trajectory.trajectory[point].velocity * mmsettings.goalVelocityWeight;
        frame.trajectory.trajectory[point].angularVelocity = frame.trajectory.trajectory[point].angularVelocity * mmsettings.goalAngularVelocityWeight;
      }
    }
    modifiedClips.push_back(clip);
  }
}

void AnimationDataBase::normalize_database(const MotionMatchingSettings &mmsettings)
{
  int frameIdx = 0;
  for (uint nextClip = 0; nextClip < clips.size(); nextClip++)
  {
    bounding.clipsStarts.push_back(frameIdx);
    AnimationClip& clip = clips[nextClip];
    for (uint nextCadr = 0, n = clip.duration; nextCadr < n; nextCadr++)
    {
      normalizedFeatures.emplace_back();
      clip.features[nextCadr].save_to_array(normalizedFeatures[frameIdx]);
      frameIdx++;
    }
  }
  const int vec_size = 3;
  std::vector<int> featuresSizes;
  FrameFeature::get_sizes(featuresSizes);
  int idx = 0, offset;
  offset = (vec_size * 2) * (int)AnimationFeaturesNode::LeftHand;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.LeftHand);
  offset += vec_size;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.LeftHandSpeed);
  offset = (vec_size * 2) * (int)AnimationFeaturesNode::RightHand;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.RightHand);
  offset += vec_size;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.RightHandSpeed);
  offset = (vec_size * 2) * (int)AnimationFeaturesNode::LeftToeBase;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.LeftToeBase);
  offset += vec_size;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.LeftToeBaseSpeed);
  offset = (vec_size * 2) * (int)AnimationFeaturesNode::RightToeBase;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.RightToeBase);
  offset += vec_size;
  normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.RightToeBaseSpeed);
  offset = (int)AnimationFeaturesNode::Count * 2 * 3;
  for (int i = 0; i < AnimationTrajectory::PathLength; ++i)
  {
    normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.goalPathMatchingWeight);
    offset += vec_size;
    normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, vec_size, mmsettings.goalVelocityWeight);
    offset += vec_size;
    normalize_feature(normalizedFeatures, bounding.featuresScale, bounding.featuresMean, offset, 1, mmsettings.goalAngularVelocityWeight);
    offset += 1;
  }
}

void AnimationDataBase::acceleration_structs(bool applySettingsOnce, bool check_existance)
{
  if (check_existance)
  {
    if (!vpTrees.empty() && !coverTrees.empty() && !kdTrees.empty() && !normalizedFeatures.empty())
      return;
  }
  std::vector<AnimationClip>& actualClips = (applySettingsOnce && !modifiedClips.empty()) ? modifiedClips : clips;
  debug_log((applySettingsOnce && !modifiedClips.empty()) ? "m" : "nm");
  vpTrees.clear();
  coverTrees.clear();
  kdTrees.clear();
  if (ecs::get_singleton<SettingsContainer>().motionMatchingSettings.empty())
    return;
  TimeScope scope("Creating acceleration structs");
  map<Tag, size_t> tagMap;
  vector<AnimationTags> treeTags;
  for (const AnimationClip &clip : actualClips)
  {
    auto it = tagMap.find(clip.tags.tags);
    if (it == tagMap.end())
    {
      tagMap.try_emplace(clip.tags.tags, tagMap.size());
      treeTags.emplace_back(clip.tags);
    }
  }
  vector<vector<VPTree::Node>> nodes(tagMap.size());
  vector<vector<CoverTree::Node>> nodes2(tagMap.size());
  vector<vector<KdTree::Node>> nodes3(tagMap.size());

  for (uint i = 0; i < actualClips.size(); i++)
  {
    size_t j = tagMap[actualClips[i].tags.tags];
    for (uint k = 0; k < actualClips[i].features.size(); k++)
    {
      nodes[j].emplace_back(VPTree::Node{&actualClips[i].features[k], i, k, 0.f, 0.f});
      nodes2[j].emplace_back(CoverTree::Node{{}, &actualClips[i].features[k], i, k, 0.f, 0.f});
      nodes3[j].emplace_back(KdTree::Node{&actualClips[i].features[k], i, k});
    }
  }
  vpTrees.reserve(nodes.size());
  const auto &settings = ecs::get_singleton<SettingsContainer>().motionMatchingSettings[0].second;
  auto f = [&](const FrameFeature &a, const FrameFeature &b)
  {
    return get_score(a, b, settings).full_score;
  };
  for (uint i = 0; i < nodes.size(); i++)
  {
    vpTrees.emplace_back(treeTags[i], std::move(nodes[i]), f);
    coverTrees.emplace_back( treeTags[i], std::move(nodes2[i]), f);
    kdTrees.emplace_back(settings, treeTags[i], std::move(nodes3[i]), f);
  }

  normalize_database(settings);
  bounding.find_boxes_values(normalizedFeatures);
}

ResourceRegister<AnimationDataBase> animDataBaseReg;