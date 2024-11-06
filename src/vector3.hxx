#ifndef CASTING_CPP_VECTOR3_HXX
#define CASTING_CPP_VECTOR3_HXX

#include <cmath>

typedef struct vec3
{
    /*
        y
        |  z
        | /
        O --- x
    */
    float x;
    float y;
    float z;
} vec3;


/// Sum two vectors
/// \param a
/// \param b
/// \return a + b
static inline vec3 vec3_add(const vec3 &a, const vec3 &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

/// Subtract two vectors
/// \param a
/// \param b
/// \return a - b
static inline vec3 vec3_sub(const vec3 &a, const vec3 &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

/// Dot product of two vectors
/// \param a : vector(3)
/// \param b : vector(3)
/// \return a . b : scalar
static inline float vec3_dot(const vec3 &a, const vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Multiply a vector by a scalar
/// \param a : vector(3)
/// \param b : scalar(1)
/// \return a * b : vector(3)
static inline vec3 vec3_dot_float(const vec3 &a, float b)
{
    return {a.x * b, a.y * b, a.z * b};
}


/// Cross product of two vectors
/// \param a
/// \param b
/// \return a x b
static inline vec3 vec3_cross(const vec3 &a, const vec3 &b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

/// Normalize a vector
/// \param a : vector(3)
/// \return |a| : scalar
static inline float vec3_length(const vec3 &a)
{
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

/// Compute distance between two points
/// \param a : vector(3)
/// \param b : vector(3)
/// \return |a - b| : scalar
static inline float vec3_dist(const vec3 &dot1, const vec3 &dot2)
{
    return sqrtf(powf(dot1.x - dot2.x, 2) + powf(dot1.y - dot2.y, 2) + powf(dot1.z - dot2.z, 2));
}

#endif //CASTING_CPP_VECTOR3_HXX
