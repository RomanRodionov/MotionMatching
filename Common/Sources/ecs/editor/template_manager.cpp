#include "template.h"
#include "Serialization/serialization.h"
#include <filesystem>
namespace fs = filesystem;
namespace ecs
{ 
  
  const Template *TemplateManager::get_template_by_name(const char *name)
  {
    for (const Template *t : instance().templates)
      if (t->name == name)
        return t;
    return nullptr;
  }
  size_t TemplateInfo::serialize(std::ostream& os) const
  {
    ecs::TypeInfo &typeInfo = ecs::TypeInfo::types()[typeHash];
    size_t s = 0;
    s += write(os, name);
    s += write(os, typeInfo.name);
    s += typeInfo.serialiser(os, data);
    return s;
  }
  size_t TemplateInfo::deserialize(std::istream& is)
  {
    size_t s = 0;
    s += read(is, name);
    nameHash = HashedString(name);
    string typeName;
    s += read(is, typeName);
    typeHash = HashedString(typeName);
    ecs::TypeInfo &typeInfo = ecs::TypeInfo::types()[typeHash];
    data = typeInfo.constructor(malloc(typeInfo.sizeOf));
    s += typeInfo.deserialiser(is, data);
    return s;
  }
  size_t Template::serialize(std::ostream& os) const
  {
    size_t s = 0;
    s += write(os, types);
    s += write(os, subTemplates.size());
    for (const Template* t: subTemplates)
      s += write(os, t->name);
    return s;
  }
  size_t Template::deserialize(std::istream& is)
  {
    vector<Template*> &templates = TemplateManager::instance().templates;
    size_t s = 0;
    s += read(is, types);
    uint subTemplSize = 0;
    s += read(is, subTemplSize);
    subTemplates.reserve(subTemplSize);
    for (uint i = 0; i < subTemplSize; ++i)
    {
      string subTemplName;
      s += read(is, subTemplName);
      auto it = std::find_if(templates.begin(), templates.end(), 
        [&](const Template *other){ return subTemplName == other->name; });
      if (it != templates.end())
      {
        (*it)->parentTemplates.push_back(this);
        subTemplates.push_back(*it);
      }
      else
      {
        debug_error("Didn't exist template %s, it is subtemplate of %s", subTemplName.c_str(), name.c_str());
      }
    }
    return s;
  }
  void load_templates()
  {
    const string &path = project_resources_path("Templates");
    vector<Template*> &templates = TemplateManager::instance().templates;
    templates.clear();
    if (!fs::exists(path))
      return;
    for (const auto & entry : fs::directory_iterator(path))
    {
      if (entry.is_regular_file() && entry.path().extension() == ".tmpl")
      {
        templates.push_back(new ecs::Template(entry.path().stem().string().c_str()));
      }
    }
    for (const Template* t: templates)
    {
      load_object(*t, "Templates/" + t->name + ".tmpl");
    }
  }
  void save_templates()
  {
    const string &path = project_resources_path("Templates");
    if (!fs::exists(path))
      fs::create_directory(path);
    vector<Template*> &templates = TemplateManager::instance().templates;
    for (const Template* t: templates)
    {
      save_object(*t, "Templates/" + t->name + ".tmpl");
    }
  }
}