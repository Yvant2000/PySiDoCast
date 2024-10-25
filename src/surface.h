#ifndef CASTING_CPP_SURFACE_H
#define CASTING_CPP_SURFACE_H

#include "geometry.h"

static constinit const int ALPHA = 3;
static constinit const int RED = 2;
static constinit const int GREEN = 1;
static constinit const int BLUE = 0;

///
/// \brief A surface in the scene
struct Surface
{
    Py_buffer buffer; // The buffer of the surface
    struct pos3 pos;  // The position of the 3 points of the triangle
    PyObject *parent;  // The python object that owns this surface
    // We need to keep a reference to the parent object to prevent the buffer from being destroyed by the garbage collector
    bool del;  // If the Surface is temporary and needs to be deleted
    bool reverse;  // If the surface texture is reversed (useful for rectangles)
    float alpha;  // The alpha value of the surface
};

/// Free a surface object
/// \param surface The surface to free
static inline void free_surface(struct Surface &surface)
{
    PyBuffer_Release(&surface.buffer);
    Py_DECREF(surface.parent);
//    free(surface);
}

/// Free temporary surfaces in the list
/// \param surfaces The list of surfaces
static inline void free_temp_surfaces(std::vector<struct Surface> &surfaces)
{
    // The temporary surfaces are shuffled into the list

    size_t back = surfaces.size();

    // delete the temporary surfaces at the end of the list
    while (back > 0 and surfaces[back - 1].del)
    {
        free_surface(surfaces[back - 1]);
        back -= 1;
    }

    back -= 1;

    for (size_t front = 0; front < back; front += 1)
    {
        if (not surfaces[front].del)
        {
            continue;
        }

        free_surface(surfaces[front]);
        surfaces[front] = surfaces[back];

        back -= 1;

        // find the next non-temporary surface
        while (back > front and surfaces[back].del)
        {
            free_surface(surfaces[back]);
            back -= 1;
        }
    }

    surfaces.resize(back + 1);
}

/// Gets the color of a pixel on a surface at the given coordinates
/// \param buffer py_buffer object containing the surface datar
/// \param u baricentric coordinate of the pixel on the surface
/// \param v baricentric coordinate of the pixel on the surface
/// \return pointer to the pixel color
static inline unsigned char *get_pixel_from_buffer(const Py_buffer &buffer, float u, float v)
{
    const Py_ssize_t width = buffer.shape[0];
    const Py_ssize_t height = buffer.shape[1];

    const auto x = (Py_ssize_t) (u * (float) width);
    const auto y = (Py_ssize_t) (v * (float) height);

    long *buf = (long *) buffer.buf;
    long *pixel = buf + (y * width + x);
    return (reinterpret_cast<unsigned char *> (pixel)) - 2;
}

/// Compute the distance to the nearest surface in the given direction
/// If no intersection is found, return the maximum distance.
/// \param ray              starting point and direction
/// \param max_distance     maximum distance to search for
/// \param surfaces         list of surfaces to search in
/// \return           distance to the nearest surface
static inline float
get_closest_intersection(const pos2 &ray, float max_distance, const std::vector<struct Surface> &surfaces)
{
    float closest = max_distance;

    for (const struct Surface surface: surfaces)
    {
        float dist, u, v;
        if (not segment_triangle_intersect(ray, surface.pos, closest, &dist, &u, &v))
            continue;

        const unsigned char *const new_pixel_ptr = get_pixel_from_buffer(surface.buffer, u, v);
        if (new_pixel_ptr[ALPHA])
            closest = dist;
    }

    return closest;
}

#endif //CASTING_CPP_SURFACE_H
