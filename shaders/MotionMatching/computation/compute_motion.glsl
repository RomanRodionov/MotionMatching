#shader compute_motion

#compute_shader

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
layout(r32f, binding = 0) uniform image2D out_tex;
layout(std430, binding = 0) buffer inout_values
{
    float data_SSBO[];
};
uniform int arr_size;

shared float values[512];

void main()
{
    uint step = 512;
    uint left_border = (gl_GlobalInvocationID.x - gl_LocalInvocationID.x) * 2;
    uint base_idx = left_border + gl_LocalInvocationID.x;
    if (base_idx < arr_size) 
    {
        values[gl_LocalInvocationID.x] = data_SSBO[base_idx];
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
            float val2 = values[gl_LocalInvocationID.x + step];
            if (val2 < val1) values[gl_LocalInvocationID.x] = val2;
        }
        step /= 2;
        memoryBarrierShared();
        barrier();
    }
    data_SSBO[base_idx] = values[gl_LocalInvocationID.x];
}