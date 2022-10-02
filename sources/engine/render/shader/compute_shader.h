#pragma once
#include "common.h"
#include "3dmath.h"
#include <vector>
#include "glad/glad.h"
#include "shader_buffer.h"
#include "shader.h"

#define BAD_PROGRAM 0
struct StorageBuffer;

class ComputeShader: public Shader
{
private:
	int shaderIdx;
public:
	ComputeShader() :shaderIdx(-1){}
	ComputeShader(int shaderIdx):shaderIdx(shaderIdx){}
	ComputeShader(const std::string &shader_name, GLuint shader_program, bool compiled, bool update_list = false);
  void dispatch(vec2 work_groups) const;
};
