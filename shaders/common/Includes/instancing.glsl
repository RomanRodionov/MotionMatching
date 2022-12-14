struct Instance
{
    Material material;
    #if BONES
    mat4 Bones[BONES];
    #endif
};

layout(std430, binding = 1) readonly buffer InstanceData 
{
  Instance instances[];
};
#ifdef VS
  #define material_inst instances[gl_InstanceID].material
#elif PS
  #define material_inst instances[instanceID].material
#endif

#include transforms