#version 450
#define CS 1




layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
layout(r32f, binding = 0) uniform image2D out_tex;
uniform int arr_size;

shared float values[512];

void main()
{
    uint step = 512;
    uint left_border = (gl_GlobalInvocationID.x - gl_LocalInvocationID.x) * 2;
    uint base_idx = left_border + gl_LocalInvocationID.x
    if (base_idx < arr_size) 
    {
        values[gl_LocalInvocationID.x] = imageLoad( out_tex, ivec2(base_idx, gl_GlobalInvocationID.y) ).r;
    }
    memoryBarrierShared();
    barrier();
    if (base_idx + step < arr_size) 
    {
        float value = imageLoad( out_tex, ivec2( (left_border + gl_LocalInvocationID.x, gl_GlobalInvocationID.y) ) ).r;
        if (value < values[gl_LocalInvocationID.x]) values[gl_LocalInvocationID.x] = value;
    }
    memoryBarrierShared();
    barrier();
    while (step > 0) {
        if ((gl_LocalInvocationID.x < step) && (base_idx + step < arr_size)) 
        {
            float val1 = values[gl_LocalInvocationID.x];
            float val2 = values[gl_LocalInvocationID.x + step]
            if (val2 < val1) values[gl_LocalInvocationID.x] = val2;
        }
        memoryBarrierShared();
        barrier();
    }
    ivec2 pos = ivec2(base_idx, gl_GlobalInvocationID.y);
    imageStore( out_tex, pos, vec4( values[gl_LocalInvocationID.x], 0.0, 0.0, 0.0 ) );
}