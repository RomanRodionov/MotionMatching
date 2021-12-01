#pragma once
#include "type_info.h"
#include "type_description.h"

namespace ecs
{
  struct ComponentInstance 
  {
private:
    vector<char> instanceData;
    
    template<typename T>
    T& get_instance()
    {
      assert(instanceData.size() == sizeof(T));
      return *(T*)(instanceData.data());
    }
  public:
    const ecs::TypeInfo *typeInfo;
    string name;
    string_hash nameHash, typeNameHash;
    ComponentInstance(ComponentInstance &&) = default;
    ComponentInstance& operator=(ComponentInstance &&) = default;
    ComponentInstance(ComponentInstance const&) = default;
    ComponentInstance& operator=(ComponentInstance const&) = default;

    template<typename T, typename TT = std::remove_cvref_t<T>>
    ComponentInstance(const ecs::TypeInfo &typeInfo, const string &name, const T &instance) : 
    instanceData(sizeof(TT)), typeInfo(&typeInfo), name(name)
    , nameHash(HashedString(name)), typeNameHash(TypeDescription::hash(nameHash, typeInfo.hashId))
    {
      typeInfo.constructor(instanceData.data());
      get_instance<TT>() = instance;
    }    
    template<typename T, typename TT = std::remove_cvref_t<T>>
    ComponentInstance(const ecs::TypeInfo &typeInfo, const string &name, T &&instance) : 
    instanceData(sizeof(TT)), typeInfo(&typeInfo), name(name)
    , nameHash(HashedString(name)), typeNameHash(TypeDescription::hash(nameHash, typeInfo.hashId))
    {
      typeInfo.constructor(instanceData.data());
      get_instance<TT>() = instance;
    }
    ComponentInstance(const ecs::TypeInfo &typeInfo, const string &name) : 
    instanceData(typeInfo.sizeOf), typeInfo(&typeInfo), name(name)
    , nameHash(HashedString(name)), typeNameHash(TypeDescription::hash(nameHash, typeInfo.hashId))
    {
      typeInfo.constructor(instanceData.data());
    }
    template<typename T, typename TT = std::remove_cvref_t<T>>
    void update(const T &instance)
    {
      get_instance<TT>() = instance;
    }    
    template<typename T, typename TT = std::remove_cvref_t<T>>
    void update(T &&instance)
    {
      get_instance<TT>() = instance;
    }    
    void *get_data()
    {
      return instanceData.data();
    }   
    const void *get_data() const 
    {
      return instanceData.data();
    }    
  };

  struct ComponentInitializerList
  {
    vector<ComponentInstance> components;
    template<typename T, typename TT = std::remove_cvref_t<T>>
    void set(const char *name, T &&value)
    {
      constexpr string_hash hash = HashedString(nameOf<TT>::value);
      const auto &types = ecs::TypeInfo::types();
      auto it = types.find(hash);
      if (it != types.end())
      {
        components.emplace_back(it->second, name, value);
      }
    }
  };

  
  void patch_components(vector<ComponentInstance> &components, const vector<ComponentInstance> &patch);
}