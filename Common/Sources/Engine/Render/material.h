#pragma once
#include "common.h"
#include "Shader/storage_buffer.h"
#include "Shader/shader.h"
#include "Texture/textures.h"
#include "Engine/Resources/asset.h"
#include "Serialization/iserializable.h"
class Property final: public ISerializable
{
private:
  vec4 property;
  Asset<Texture2D> texture;
  string name;
  int vecType;
public:
  Property() = default;
  Property(const string& name, Asset<Texture2D> property)
    :property(vec4()), texture(property), name(name) {vecType = 5;}
  Property(const string& name, vec4 property)
    :property(property), texture(), name(name) {vecType = 4;}
  Property(const string& name, vec3 property)
    :Property(name, vec4(property, 0)) {vecType = 3;}
  Property(const string& name, vec2 property)
    :Property(name, vec3(property, 0)) {vecType = 2;}
  Property(const string& name, float property)
    :Property(name, vec2(property, 0)) {vecType = 1;}
  void bind_to_shader(const Shader &shader) const;
  void unbind_to_shader(const Shader &shader) const;
  bool operator== (const Property & other) const;
  static bool edit_property(Property& property);
  virtual size_t serialize(std::ostream& os) const override;
  virtual size_t deserialize(std::istream& is) override;
};


  #define TYPES \
  TYPE(float, GL_FLOAT) TYPE(vec2, GL_FLOAT_VEC2) TYPE(vec3, GL_FLOAT_VEC3) TYPE(vec4, GL_FLOAT_VEC4) \
  TYPE(int, GL_INT) TYPE(ivec2, GL_INT_VEC2) TYPE(ivec3, GL_INT_VEC3) TYPE(ivec4, GL_INT_VEC4)\
  TYPE(mat2, GL_FLOAT_MAT2) TYPE(mat3, GL_FLOAT_MAT3) TYPE(mat4, GL_FLOAT_MAT4) 
//TYPE(uint, GL_UNSIGNED_INT) TYPE(uvec2, GL_UNSIGNED_INT_VEC2) TYPE(uvec3, GL_UNSIGNED_INT_VEC3) TYPE(uvec4, GL_UNSIGNED_INT_VEC4)


class Material : IAsset
{
private:
  Shader shader;
  map<string, int> uniformMap;
  #define TYPE(T, _) vector<T> T##s;
  TYPES
  #undef TYPE
  #define TYPE(T, _) (vector<pair<string, vector<T>>>) (T##savable),
  REFLECT(Material,
  TYPES
  (vector<Property>) (properties),
  (string) (shaderName))
  #undef TYPE
  pair<int, int> get_uniform_index(const char *name, int gl_type) const;
public:
  Material() = default;
  Material(const vector<Property> & properties):
    properties(properties) { }
  Shader &get_shader();
  const Shader &get_shader() const;
  void set_property(const Property &property);
  void bind_to_shader() const;
  void unbind_to_shader() const;
  virtual void load(const filesystem::path &path, bool reload) override;
  virtual void free() override;
  virtual bool edit() override;

  uint buffer_size() const;
  void set_data_to_buf(char *data) const;

  #define TYPE(T, gl_type)\
  bool set_property(const char *name, const T &value)\
  {\
    auto [offset, size] = get_uniform_index(name, gl_type);\
    if (offset >= 0)\
      T##s[offset] = value;\
    return offset >= 0;\
  }\
  bool set_property(const char *name, const vector<T> &value)\
  {\
    auto [offset, size] = get_uniform_index(name, gl_type);\
    if (offset >= 0)\
      for (uint i = 0, n = glm::min(size, (int)value.size()); i < n; ++i)\
        T##s[offset + i] = value[i];\
    return offset >= 0;\
  }
  
  TYPES
  #undef TYPE
};
template<typename T>
class Asset;

Asset<Material> standart_material();
Asset<Material> standart_textured_material(Asset<Texture2D> texture);