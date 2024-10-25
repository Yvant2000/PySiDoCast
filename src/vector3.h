#ifndef CASTING_CPP_VECTOR3_H
#define CASTING_CPP_VECTOR3_H

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
vec3 vec3_add(const vec3 &a, const vec3 &b);

/// Subtract two vectors
/// \param a
/// \param b
/// \return a - b
vec3 vec3_sub(const vec3 &a, const vec3 &b);

/// Dot product of two vectors
/// \param a : vector(3)
/// \param b : vector(3)
/// \return a . b : scalar
float vec3_dot(const vec3 &a, const vec3 &b);

/// Multiply a vector by a scalar
/// \param a : vector(3)
/// \param b : scalar(1)
/// \return a * b : vector(3)
vec3 vec3_dot_float(const vec3 &a, float b);

/// Cross product of two vectors
/// \param a
/// \param b
/// \return a x b
vec3 vec3_cross(const vec3 &a, const vec3 &b);

/// Normalize a vector
/// \param a : vector(3)
/// \return |a| : scalar
float vec3_length(const vec3 &a);

/// Compute distance between two points
/// \param a : vector(3)
/// \param b : vector(3)
/// \return |a - b| : scalar
float vec3_dist(const vec3 &dot1, const vec3 &dot2);

#endif //CASTING_CPP_VECTOR3_H
