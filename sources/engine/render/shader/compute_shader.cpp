#include "shader.h"
#include "compute_shader.h"
#include <iostream>
#include <map>
#include "shader_gen.h"
#include "type_registration.h"

ECS_REGISTER_TYPE(compute_shader, ComputeShader, true, true);

void read_shader_info(const std::string &, ShaderInfo &shader);

static std::vector<std::pair<std::string, ShaderInfo>> shaderList;
static Shader badShader(-1);


ComputeShader::ComputeShader(const std::string &shader_name, GLuint shader_program, bool compiled, bool update_list)
{
  if (shader_program == BAD_PROGRAM)
    return;
  for (shaderIdx = 0; shaderIdx < (int)shaderList.size() && shaderList[shaderIdx].first != shader_name; ++shaderIdx);

  if (shaderIdx >= (int)shaderList.size())
  {
    shaderList.emplace_back(shader_name, ShaderInfo{shader_program, compiled, {}});
    read_shader_info(shaderList.back().first, shaderList.back().second);
  }
  else
  if (update_list)
  {
    glDeleteProgram(shaderList[shaderIdx].second.program);
    shaderList[shaderIdx] = make_pair(shader_name, ShaderInfo{shader_program, compiled, {}});
    read_shader_info(shaderList[shaderIdx].first, shaderList[shaderIdx].second);
  }
}
