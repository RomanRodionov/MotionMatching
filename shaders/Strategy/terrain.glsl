#shader terrain_shader

#define USE_UV

struct Material
{
  #define LAMBERT
  #include material
  float atlasScale;
  float colorMapWeigth;
  float seasonTime;
  ivec2 mapSize;
  vec2 texelSize;
};
#include vs_output
#include instancing
#include common_data

#vertex_shader

#include vs_input
uniform sampler2D heightMap;
void main()
{
  vec3 vertex_position = Position;
  float height = texture(heightMap, TexCoord).y;
  vertex_position.y += height * 1.0;
  #include common_vs
}

#pixel_shader

#include ps_input
out vec4 FragColor;

uniform sampler2D heightMap;
uniform sampler2D provincesMap;
uniform sampler2D normalMap;
uniform usampler2D terrainMap;
uniform sampler2DArray terrainDiffuseArray;
uniform sampler2DArray terrainNormalArray;
uniform sampler2DArray terrainColormapArray;

#include lambert_lighting
#include normal_map

uint terrain_type(vec2 physUV, vec2 detailedUV, ivec2 mapSize, out vec4 finalColor, out vec3 finalNormal)
{
  const int N = 4;
  vec4 color[N];
  vec3 normal[N];
  uvec4 terrain = textureGather(terrainMap, physUV, 0);
  for (int i = 0; i < N; i++)
  {
    uint type = terrain[i];
    vec3 atlasUV = vec3(detailedUV, type);
    if (type <= 10)
    {
      color[i] = vec4(texture(terrainDiffuseArray, atlasUV).rgb, 1);
      normal[i] = texture(terrainNormalArray, atlasUV).xyz * 2.0 - 1.0;
    }
    else
    {
      color[i] = vec4(0,0,0,0);
      normal[i] = vec3(0,1,0);
    }
  }
  vec2 uv = physUV * mapSize - 0.5;
  vec2 a = fract(uv);
  vec2 b = 1 - a;
  float w[N] = float[](b.x*a.y, a.x*a.y, a.x*b.y, b.x*b.y);

  for (int i = 0; i < N; i++)
  {
    finalColor += color[i] * w[i];
    finalNormal += normal[i] * w[i];
  }
  finalNormal = normalize(finalNormal);
  return textureLod(terrainMap, physUV, 0).x;
}
  

void main()
{
  vec4 tex = texture(provincesMap, vsOutput.UV);

  float yearTime = Time.x * material_inst.seasonTime;
  int season = int(yearTime) & 3;
  int next = (season+1) & 3;
  float progress = fract(yearTime);
  vec3 colorMap = 
    mix(texture(terrainColormapArray, vec3(vsOutput.UV, season)).rgb,
        texture(terrainColormapArray, vec3(vsOutput.UV, next)).rgb,
        progress);

  
  vec3 texColor = colorMap;
  vec3 normalMap = texture(normalMap, vsOutput.UV).xyz * 2.0 - 1.0;
  vec3 normal = vsOutput.EyespaceNormal;
  normal = apply_normal_map(normalMap, vsOutput.EyespaceNormal, vsOutput.WorldPosition, vsOutput.UV);

  vec4 detailColor = vec4(0,0,0,0);
  vec3 detailNormal = vec3(0,0,1);
  vec2 detailedUV = vsOutput.UV * material_inst.atlasScale;
  uint terrainType = terrain_type(vsOutput.UV, detailedUV, material_inst.mapSize, detailColor, detailNormal);
  
  if (terrainType <= 10)
  {
    normal = apply_normal_map(detailNormal, normal, vsOutput.WorldPosition, detailedUV);
    texColor = mix(detailColor.rgb, texColor, material_inst.colorMapWeigth);
  }

  vec3 color = LightedColor(texColor, material_inst, vsOutput.WorldPosition, normal, LightDirection, CameraPosition);
  FragColor = vec4(color, 1.0);
}