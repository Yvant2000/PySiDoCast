#include "vector3.h"

#include <cmath>

vec3 vec3_add(const vec3 &a, const vec3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

vec3 vec3_sub(const vec3 &a, const vec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float vec3_dot(const vec3 &a, const vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

 vec3 vec3_dot_float(const vec3 &a, float b)
{
    return {a.x * b, a.y * b, a.z * b};
}

vec3 vec3_cross(const vec3 &a, const vec3 &b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float vec3_length(const vec3 &a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

float vec3_dist(const vec3 &dot1, const vec3 &dot2)
{
    return sqrtf(powf(dot1.x - dot2.x, 2) + powf(dot1.y - dot2.y, 2) + powf(dot1.z - dot2.z, 2));
}
