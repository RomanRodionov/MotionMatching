#include "ecs/ecs.h"
#include "common.h"
#include "Engine/imgui/imgui.h"
#include "Engine/Resources/resources.h"
#include "ecs/singleton.h"

struct SelectedAsset : ecs::Singleton
{
  Asset<AssetStub> *asset = nullptr;
  string_view name;
  ResourceMap *resourceType = nullptr;
};
SYSTEM(ecs::SystemOrder::UIMENU, ecs::SystemTag::Editor) resources_menu(SelectedAsset &selectedAsset)
{
  if (ImGui::BeginMenu("Resources"))
  {
    auto & assets = Resources::instance().assets;
    for (auto &p : assets)
    {
      if (ImGui::Button(p.first.data()))
      {
        selectedAsset.resourceType = &p.second;
        selectedAsset.name = p.first;
      }
    }
    ImGui::EndMenu();
  }
}

SYSTEM(ecs::SystemOrder::UI, ecs::SystemTag::Editor) asset_viewer(SelectedAsset &selectedAsset)
{
  constexpr int BUFN = 255;
  char buf[BUFN];
  if (selectedAsset.resourceType)
  {
    if (ImGui::Begin("asset viewer"))
    {
      static bool adding = false;
      ImGui::Text("%s", selectedAsset.name.data());
      if (!selectedAsset.resourceType->metaDataAsset)
      {
        if (adding)
        {
          static string bufString = "";
          snprintf(buf, BUFN, "%s", bufString.c_str());
          if (ImGui::InputText("", buf, BUFN))
          {
            bufString.assign(buf);
          }
            
          if (bufString == "")
            ImGui::TextColored(ImVec4(1,0,0,1), "Enter name");
          else
          {
            auto it = selectedAsset.resourceType->resources.find(bufString);
            if (it != selectedAsset.resourceType->resources.end())
            {
              ImGui::TextColored(ImVec4(1,0,0,1), "There is asset with this name");
            }
            else
            {
              static bool wantCopy = false;
              static Asset<AssetStub> stub;
              static const char *copyName;
              ImGui::SameLine();
              if (ImGui::Button("Copy"))
                wantCopy = !wantCopy, stub = Asset<AssetStub>();
              if (wantCopy)
              {
                static string searchString = "";
                snprintf(buf, BUFN, "%s", searchString.c_str());
                if (ImGui::InputText("Search substr", buf, BUFN))
                {
                  searchString.assign(buf);
                }
                static int curCopy = -1;
                vector<const char *> names;
                
                for (auto &asset : selectedAsset.resourceType->resources)
                  if (strstr(asset.first.c_str(), searchString.c_str()))
                    names.push_back(asset.first.c_str());
                if (ImGui::ListBox("", &curCopy, names.data(), names.size()))
                {
                  stub = selectedAsset.resourceType->resources[string(names[curCopy])];
                  copyName = names[curCopy];
                }
              }
              ImGui::SameLine();
              snprintf(buf, BUFN, "%s%s", stub ? "Copy " : "Create", stub ?  copyName : "");
              if (ImGui::Button(buf))
              {
                string folder = project_resources_path(selectedAsset.resourceType->name);
                if (!filesystem::exists(folder))
                  filesystem::create_directory(folder);
                if (stub)
                  selectedAsset.resourceType->createCopyAsset(folder + "\\" + bufString + "." + selectedAsset.resourceType->name, stub);
                else
                  selectedAsset.resourceType->createAsset(folder + "\\" + bufString + "." + selectedAsset.resourceType->name);
                adding = false;
                bufString = "";
              }
              
            }
          }
          if (ImGui::Button("Back"))
            adding = false;
        }
        else
        {
          ImGui::SameLine();
          if (ImGui::Button("Add asset"))
          {
            adding = true;
          }
        }
      }
      ImGui::SameLine();
      if (ImGui::Button("Close viewer"))
      {
        selectedAsset.resourceType = nullptr;
        selectedAsset.asset = nullptr;
        adding = false;
        goto end;
      }
      if (!adding)
      {
        if (selectedAsset.asset)
        {
          if (ImGui::Button("Back"))
            selectedAsset.asset = nullptr;
          else
          if (selectedAsset.asset->loaded() && selectedAsset.resourceType->editAsset(*selectedAsset.asset))
            selectedAsset.resourceType->reloadAsset(*selectedAsset.asset);
        }
        else
        for (auto &asset : selectedAsset.resourceType->resources)
        {
          if (ImGui::Button(asset.first.c_str()))
          {
            selectedAsset.asset = &asset.second;
            selectedAsset.resourceType->loadAsset(asset.second, false);
          }
        }
      }
    }
end:
    ImGui::End();
  }
}
