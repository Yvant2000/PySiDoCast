#ifndef CASTING_CPP_GEOMETRY_H
#define CASTING_CPP_GEOMETRY_H

#include "vector3.h"

struct pos2
{
    /* For a segment, the start point and end point.
        A [---------] B
    */
    /* For a line, a point and a direction.
        ------ A ------ (B) ->
    */
    const vec3 A;
    const vec3 B;
};

struct pos3
{
    /*
        C
        |\
        | \
        |  \
        |   \
        |    \
        A --- B
     */
    vec3 A;
    vec3 B;
    vec3 C;
};

/// When the function returns true, the intersection point is given by R.Origin + t * R.Dir
/// The barycentric coordinates of the intersection in the triangle are u, v, 1-u-v (useful for Gouraud shading or texture mapping)
/// \param segment  The segment to test
/// \param triangle  The triangle to intersect
/// \param closest  The closest intersection point already found
/// \param dist the distance from the origin of the ray to the intersection point
/// \param u baricentric coordinate
/// \param v baricentric coordinate
/// \return true if the ray intersects the triangle, false otherwise
static inline bool
segment_triangle_intersect(const pos2 &segment, const pos3 &triangle, float closest, float *dist, float *u, float *v)
{
    vec3 E1 = vec3_sub(triangle.B, triangle.A);
    vec3 E2 = vec3_sub(triangle.C, triangle.A);
    vec3 N = vec3_cross(E1, E2);

//    vec3 N = {(triangle.B.y - triangle.A.y) * (triangle.C.z - triangle.A.z) - (triangle.B.z - triangle.A.z) * (triangle.C.y - triangle.A.y),
//              (triangle.B.z - triangle.A.z) * (triangle.C.x - triangle.A.x) - (triangle.B.x - triangle.A.x) * (triangle.C.z - triangle.A.z),
//              (triangle.B.x - triangle.A.x) * (triangle.C.y - triangle.A.y) - (triangle.B.y - triangle.A.y) * (triangle.C.x - triangle.A.x)};

    float det = -vec3_dot(segment.B, N);

    vec3 AO = vec3_sub(segment.A, triangle.A);
    *dist = vec3_dot(AO, N) / det;

    if (*dist < 0 || *dist >= closest)  // The test "*dist < 0" prevent the camera to enter the dark mirror dimension
        return false;

    vec3 DAO = vec3_cross(AO, segment.B);

    *u = vec3_dot(E2, DAO) / det;
    if (*u < 0)  // prevent the surfaces from being stretched to infinity (causes a crash)
        return false;

    *v = -vec3_dot(E1, DAO) / det;

    return (*v >= 0. && (*u + *v) <= 1.0);  // prevent the surfaces from being stretched to infinity (causes a crash)
    // -vec3_dot(dir, N) >= EPSILON // prevent the surfaces from being seen from behind
}

/// Compute the minimum distance between a point and a line
/// \param point position of the point
/// \param line_point position of a point on the line
/// \param line_direction vector representing the direction of the line
/// \return distance
static inline float line_point_distance(const vec3 point, const vec3 line_point, const vec3 line_direction)
{
    const vec3 s = vec3_sub(line_direction, line_point);
    const vec3 w = vec3_sub(point, line_point);
    const float ps = vec3_dot(w, s);

    if (ps <= 0)
        return vec3_length(w);

    const float l2 = vec3_dot(s, s);
    if (ps >= l2)
        return vec3_length(vec3_sub(point, line_direction));

    return vec3_length(vec3_sub(point, vec3_add(line_point, vec3_dot_float(s, ps / l2))));
}

#endif //CASTING_CPP_GEOMETRY_H
