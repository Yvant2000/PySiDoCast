#include <Python.h>

#define _USE_MATH_DEFINES

#include <cmath>
#include <mutex>
#include <thread>
#include <queue>

#include "dithering.h"
#include "vector3.h"

static constinit const int ALPHA = 3;
static constinit const int RED = 2;
static constinit const int GREEN = 1;
static constinit const int BLUE = 0;


/// PyObject containing the RayCaster
typedef struct t_RayCasterObject
{
    PyObject_HEAD           // required python object header
    std::vector<struct Surface> surfaces;     // List of surfaces
    std::vector<struct Light> lights;
} RayCasterObject;

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

// TODO: measure is pass by ref is faster
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


///
/// \brief A light source in the scene.
struct Light
{
    const vec3 pos;  // Position of the light in the scene
    const vec3 direction;  // Direction of the light in the scene
    const float pos_direction_distance;  // Distance between the light and the direction
    const float intensity; // Intensity of the light (= the distance of lightning)
    const float r; // Red component of the light
    const float g; // Green component of the light
    const float b; // Blue component of the light
    // TODO: Might add an intensity offset (useful for cel shading)
};


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

/// gets the py_buffer from a pygame surface
/// \attention For this to work, you must use `.convert_alpha()` on the surface before passing it to this function
/// \param img          pygame surface
/// \param buffer       the buffer from the image
/// \return         true on error, false on success
static inline bool _get_3DBuffer_from_Surface(PyObject * img, Py_buffer * buffer)
{
    PyObject * get_view_method = PyObject_GetAttrString(img, "get_view");
    if (get_view_method == nullptr)
    {
        printf("Error: Could not get the get_view method from the surface\n");
        return true;
    }

    PyObject * arg = Py_BuildValue("y", "3");
    PyObject * view = PyObject_CallOneArg(get_view_method, arg); // array of width * height * RGBA

    Py_DECREF(arg);
    Py_DECREF(get_view_method);

    if (PyObject_GetBuffer(view, buffer, PyBUF_STRIDES) == -1)
    {
        Py_DECREF(view);
        printf("Error: Could not get the buffer from the view\n");
        return true;
    }

    Py_DECREF(view);

    return false;
}

static inline int _get_float_from_tuple(int index, PyObject *tuple, float &result)
{
    PyObject * arg = PyLong_FromLong(index);
    PyObject * item;
    if ((item = PyObject_GetItem(tuple, arg)) == nullptr)
    {
        printf("Can't access index %d\n", index);
        Py_DECREF(arg);
        return -1;
    }

    result = static_cast<float>(PyFloat_AsDouble(item));

    Py_DECREF(arg);
    Py_DECREF(item);
    if (PyErr_Occurred())
    {
        printf("Error: Could not convert item %d to float", index);
        return -1;
    }

    return 0;
}

/// Gets a vec3 from a tuple
/// \param tuple    the tuple
/// \param v        pointer where the result will be stored
/// \return         0 on success, -1 on error
inline int _get_vec3_from_tuple(PyObject * tuple, vec3 &v)
{
    if (_get_float_from_tuple(0, tuple, v.x)
        || _get_float_from_tuple(1, tuple, v.y)
        || _get_float_from_tuple(2, tuple, v.z))
        return -1;
    return 0;
}

/// Add a triangle to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_triangle(RayCasterObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * surface_image;

    PyObject * py_A;
    PyObject * py_B;
    PyObject * py_C;

    float alpha = 1.0f;

    bool del = false;
    bool reverse = false;

    static char *kwlist[] = {"image", "A", "B", "C", "alpha", "rm", "reverse", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|fpp", kwlist,
                                     &surface_image, &py_A, &py_B, &py_C, &alpha, &del, &reverse))
        return nullptr;

    vec3 A;
    vec3 B;
    vec3 C;

    if (_get_vec3_from_tuple(py_A, A)
        || _get_vec3_from_tuple(py_B, B)
        || _get_vec3_from_tuple(py_C, C))
    {
        return nullptr;
    }

    Py_buffer buffer;
    if (_get_3DBuffer_from_Surface(surface_image, &buffer))
    {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        return nullptr;
    }

    struct Surface surface{
            .buffer = buffer,
            .pos = {A, B, C,},
            .parent = surface_image,
            .del = del,
            .reverse = reverse,
            .alpha = alpha,
    };

    self->surfaces.emplace_back(surface);

    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.

    Py_RETURN_NONE;
}

/// Add a surface (two triangles) to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_surface(RayCasterObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * surface_image;

    PyObject * py_A;
    PyObject * py_B;
    PyObject * py_C;

    float alpha = 1.0f;

    bool del = false;

    static char *kwlist[] = {"image", "A", "B", "C", "alpha", "rm", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|fp", kwlist,
                                     &surface_image, &py_A, &py_B, &py_C, &alpha, &del))
        return nullptr;

    vec3 A;
    vec3 B;
    vec3 C;

    if (_get_vec3_from_tuple(py_A, A)
        || _get_vec3_from_tuple(py_B, B)
        || _get_vec3_from_tuple(py_C, C))
    {
        return nullptr;
    }

    Py_buffer buffer1;
    Py_buffer buffer2;

    if (_get_3DBuffer_from_Surface(surface_image, &buffer1)
        || _get_3DBuffer_from_Surface(surface_image, &buffer2))
    {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        return nullptr;
    }

    struct Surface surface{
            .buffer = buffer1,
            .pos = {A, B, C},
            .parent = surface_image,
            .del = del,
            .reverse = false,
            .alpha = alpha,
    };

    struct Surface surface2{
            .buffer = buffer2,
            .pos = {vec3_sub(vec3_add(C, B), A), C, B},
            .parent = surface_image,
            .del = del,
            .reverse = true,
            .alpha = alpha,
    };

    self->surfaces.emplace_back(surface);
    self->surfaces.emplace_back(surface2);

    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.
    Py_INCREF(surface_image); // Two surfaces means we need to incref twice

    Py_RETURN_NONE;
}


/// Add a quad (two triangles) to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_quad(RayCasterObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * surface_image;

    PyObject * py_A;
    PyObject * py_B;
    PyObject * py_C;
    PyObject * py_D;

    float alpha = 1.0f;

    bool del = false;

    static char *kwlist[] = {"image", "A", "B", "C", "D", "alpha", "rm", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|fp", kwlist,
                                     &surface_image, &py_A, &py_B, &py_C, &py_D, &alpha, &del))
        return nullptr;

    vec3 A;
    vec3 B;
    vec3 C;
    vec3 D;

    if (_get_vec3_from_tuple(py_A, A)
        || _get_vec3_from_tuple(py_B, B)
        || _get_vec3_from_tuple(py_C, C)
        || _get_vec3_from_tuple(py_D, D))
    {
        return nullptr;
    }

    Py_buffer buffer1;
    Py_buffer buffer2;

    if (_get_3DBuffer_from_Surface(surface_image, &buffer1)
        || _get_3DBuffer_from_Surface(surface_image, &buffer2))
    {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        return nullptr;
    }

    struct Surface surface{
            .buffer = buffer1,
            .pos = {A, B, D},
            .parent = surface_image,
            .del = del,
            .reverse = false,
            .alpha = alpha,
    };

    struct Surface surface2{
            .buffer = buffer2,
            .pos = {C, D, B},
            .parent = surface_image,
            .del = del,
            .reverse = true,
            .alpha = alpha,
    };

    self->surfaces.emplace_back(surface);
    self->surfaces.emplace_back(surface2);

    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.
    Py_INCREF(surface_image); // Two surfaces means we need to incref twice

    Py_RETURN_NONE;
}


/// Add a wall (two triangles) to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_wall(RayCasterObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * surface_image;

    PyObject * py_A;
    PyObject * py_B;

    float alpha = 1.0f;

    bool del = false;

    static char *kwlist[] = {"image", "A", "B", "alpha", "rm", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|fp", kwlist,
                                     &surface_image, &py_A, &py_B, &alpha, &del))
        return nullptr;

    vec3 A;
    vec3 B;

    if (_get_vec3_from_tuple(py_A, A)
        || _get_vec3_from_tuple(py_B, B))
    {
        return nullptr;
    }

    struct Surface &surface = self->surfaces.emplace_back();

    surface.pos = {A, {B.x, A.y, B.z}, {A.x, B.y, A.z}};
    surface.parent = surface_image;
    surface.del = del;
    surface.reverse = false;
    surface.alpha = alpha;

    struct Surface &surface2 = self->surfaces.emplace_back();

    surface2.pos = {B, {A.x, B.y, A.z}, {B.x, A.y, B.z}};
    surface2.parent = surface_image;
    surface2.del = del;
    surface2.reverse = true;
    surface2.alpha = alpha;

    if (_get_3DBuffer_from_Surface(surface_image, &surface.buffer) ||
        _get_3DBuffer_from_Surface(surface_image, &surface2.buffer))
    {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        self->surfaces.pop_back();
        self->surfaces.pop_back();
        return nullptr;
    }

    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.
    Py_INCREF(surface_image); // Two surfaces means we need to incref twice

    Py_RETURN_NONE;
}


static PyObject *method_add_light(RayCasterObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * py_light;

    float light_intensity = 1.0;

    float red = 1.0;  // white by default
    float green = 1.0;
    float blue = 1.0;

    PyObject * py_direction = nullptr;

    static char *kwlist[] = {"position", "intensity", "red", "green", "blue", "direction", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ffffO", kwlist,
                                     &py_light, &light_intensity, &red, &green, &blue, &py_direction))
        return nullptr;

    vec3 light_pos;
    if (_get_vec3_from_tuple(py_light, light_pos))
    {
        return nullptr;
    }

    vec3 light_dir = {FP_NAN, FP_NAN, FP_NAN};
    if (py_direction)
        if (_get_vec3_from_tuple(py_direction, light_dir))
            return nullptr;


    if (red > 1.0f)  // clamp the color values
        red = 1.0f;
    if (green > 1.0f)
        green = 1.0f;
    if (blue > 1.0f)
        blue = 1.0f;

//    // test if one of the direction value is set but not the other
//    if (isnan(direction_x) != isnan(direction_y) || isnan(direction_x) != isnan(direction_z)) {
//        PyErr_SetString(PyExc_ValueError, "Directional light must have all direction values set or none");
//        return nullptr;
//    }

    struct Light light = {
            light_pos,
            light_dir,
            py_direction ? vec3_dist(light_pos, light_dir) : FP_NAN,
            light_intensity,
            red,
            green,
            blue};

    self->lights.emplace_back(light);

    Py_RETURN_NONE;
}


/// Free all the surfaces in the raycaster
/// \param self  The raycaster object
/// \return (Python) None
static PyObject *method_clear_surfaces(RayCasterObject *self)
{
    for (struct Surface &surface: self->surfaces)
    {
        free_surface(surface);
    }

    self->surfaces.clear();

    Py_RETURN_NONE;
}


/// Free all the lights in the raycaster
/// \param self  The raycaster object
/// \return (Python) None
static PyObject *method_clear_lights(RayCasterObject *self)
{
    self->lights.clear();
    Py_RETURN_NONE;
}


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

/// Compute weather or not a pixel should be drawn depending of the alpha value
/// \param alpha            alpha value of the pixel
/// \param x                x coordinate of the pixel on the screen
/// \param y                y coordinate of the pixel on the screen
/// \return         true if the pixel should be skipped, false otherwise
static inline bool alpha_dither(float alpha, Py_ssize_t x, Py_ssize_t y)
{
    if (alpha == 0.0f)  // pixel isn't seen at all
        return true;

    if (alpha == 1.0f)  // pixel isn't transparent
        return false;

    return alpha < DITHER_MATRIX[(y % DITHERING_SIZE) * DITHERING_SIZE + (x % DITHERING_SIZE)];
}


/// Compute the intersection between a ray and the surfaces in the raycaster and return the color of the pixel
/// \param surfaces surfaces in the scene
/// \param ray ray to cast
/// \param view_distance render distance
/// \return pixel color
static inline long
get_pixel_at(const RayCasterObject *raycaster, const struct pos2 &ray, Py_ssize_t pixel_index_x,
             Py_ssize_t pixel_index_y, float view_distance)
{
    float closest = view_distance;
    const unsigned char *closest_pixel_ptr = nullptr;

    for (const struct Surface surface: raycaster->surfaces)
    {

        if (alpha_dither(surface.alpha, pixel_index_x, pixel_index_y))
            continue;

        float dist, u, v;
        if (!segment_triangle_intersect(ray, surface.pos, closest, &dist, &u, &v))
            continue;

        if (surface.reverse)
        {  // reverse the texture if needed
            u = 1.0f - u;
            v = 1.0f - v;
        }

        const unsigned char *const new_pixel_ptr = get_pixel_from_buffer(surface.buffer, u, v);
        if (alpha_dither((float) new_pixel_ptr[ALPHA] / 255.f, pixel_index_x, pixel_index_y))
            continue;

        closest = dist;
        closest_pixel_ptr = new_pixel_ptr;
    }

    if (closest_pixel_ptr == nullptr)
        return 0;

    long pixel = 0;
    auto *pixel_ptr = reinterpret_cast<unsigned char *>(&pixel);

    const float r = 1.0f - (closest / view_distance);
    pixel_ptr[BLUE] = static_cast<unsigned char>(static_cast<float>(closest_pixel_ptr[BLUE]) * r);
    pixel_ptr[GREEN] = static_cast<unsigned char> (static_cast<float>(closest_pixel_ptr[GREEN]) * r);
    pixel_ptr[RED] = static_cast<unsigned char>(static_cast<float>(closest_pixel_ptr[RED]) * r);
    // pixel_ptr[ALPHA] = new_pixel_ptr[ALPHA];

    if (raycaster->lights.empty() || not pixel)
        return pixel;

    // apply lights

    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;

    // position in space of the pixel
    const vec3 inter = vec3_add(ray.A, vec3_dot_float(ray.B, closest)); // inter = A + t * B

    for (const struct Light &temp_light: raycaster->lights)
    {  // iterate over lights
        const float dist1 = vec3_dist(temp_light.pos, inter);  // distance between the light and the intersection
        const float ratio = temp_light.pos_direction_distance == FP_NAN
                            // if the light is a radial light, calculate the ratio
                            ? dist1 / temp_light.intensity
                            // if the light is a directional light, calculate the ratio
                            : line_point_distance(inter, temp_light.pos, temp_light.direction) *
                              temp_light.pos_direction_distance / (dist1 * temp_light.intensity);

        if (ratio < 1.0f)
        {  // ratio > 1 means the light is too far away, we don't see anything
            const float temp = 1.0f - ratio;
            red += temp * temp_light.r;
            green += temp * temp_light.g;
            blue += temp * temp_light.b;
        }
    }
    // Prevent the pixel from being too bright
    if (red > 1.0f)
        red = 1.0f;
    if (green > 1.0f)
        green = 1.0f;
    if (blue > 1.0f)
        blue = 1.0f;

    pixel_ptr[BLUE] = static_cast<unsigned char>(static_cast<float> (pixel_ptr[BLUE]) * blue);
    pixel_ptr[GREEN] = static_cast<unsigned char>(static_cast<float> (pixel_ptr[GREEN]) * green);
    pixel_ptr[RED] = static_cast<unsigned char>(static_cast<float>  (pixel_ptr[RED]) * red);

    return pixel;
}

/// data for each individual thread
struct thread_args
{          // a few args the thread needs to compute the pixel
    unsigned long *buf;      // where to write the pixel
    Py_ssize_t pixel_index;  // index of the pixel
    vec3 proj;               // projection of the pixel
};


static std::mutex queue_mutex;  // Allows only one thread to access the queue at a time
static std::queue<struct thread_args> args_queue;  // Queue of data that is NOT shared between threads
static bool thread_quit;   // Tells the threads to quit once they are done with their current task

// SHARED DATA
// (not so pretty, but it works)
// I store here all the data and shit that will be shared across all the threads

static const RayCasterObject *t_raycaster;  // Raycaster object
static float t_view_distance;  // View distance of the current scene
static Py_ssize_t t_width;  // width of the current screen (for some reason we don't need the height)

static vec3 t_width_vector;  // vector from the top right corner of the screen to the top left corner divided by the width of the screen
static struct vec3 t_A;    // This is the position of the camera

void thread_worker()
{
    while (true)  // while the job isn't done
    {
        queue_mutex.lock();  // wait until I get the lock
        if (args_queue.empty())  // check if there is job to do
        {
            queue_mutex.unlock(); // if not release the lock
            if (thread_quit)  // if the job is done and we need to quit
                return;

            std::this_thread::sleep_for(
                    std::chrono::microseconds(1));  // wait a bit before checking again to not waste cpu
            continue;
        }

        const struct thread_args args = args_queue.front();  // get my args from the queue
        args_queue.pop();
        queue_mutex.unlock();  // release the lock so other threads can get their args

        unsigned long *buf = args.buf;
        const Py_ssize_t pixel_index_y = args.pixel_index;
        vec3 ray_b = args.proj;

        for (Py_ssize_t dst_x = t_width; dst_x; --dst_x)
        {
            const struct pos2 ray{
                    .A = t_A,
                    .B = ray_b,
            };

            const long pixel = get_pixel_at(t_raycaster, ray, dst_x, pixel_index_y, t_view_distance);

            if (pixel)
                *buf = pixel;

            ray_b = vec3_add(ray_b, t_width_vector);

            buf += 1;
        }
    }
}


static PyObject *method_raycasting(RayCasterObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * screen;

    PyObject * py_pos;

    float angle_x = 0.f;
    float angle_y = 0.f;

    float fov = 70.f;
    float view_distance = 1000.f;
    bool rad = false;

    int thread_count = 1;

    static char *kwlist[] = {"dst_surface", "pos", "angle_x", "angle_y", "fov", "view_distance", "rad", "threads",
                             nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ffffpi", kwlist,
                                     &screen, &py_pos, &angle_x, &angle_y, &fov, &view_distance, &rad, &thread_count))
        return nullptr;

    vec3 pos;
    if (_get_vec3_from_tuple(py_pos, pos))
        return nullptr;


    if (fov <= 0.f)
    {
        PyErr_SetString(PyExc_ValueError, "fov must be greater than 0");
        return nullptr;
    }
    if (view_distance <= 0.f)
    {
        PyErr_SetString(PyExc_ValueError, "view_distance must be greater than 0");
        return nullptr;
    }

    if (thread_count == 0)
    {
        PyErr_SetString(PyExc_ValueError, "thread_count can't be 0");
        return nullptr;
    } else if (thread_count < 0)
        thread_count = (int) std::thread::hardware_concurrency();


    Py_buffer dst_buffer;
    if (_get_3DBuffer_from_Surface(screen, &dst_buffer))
    {
        PyErr_SetString(PyExc_ValueError, "dst_surface is not a valid surface");
        return nullptr;
    }

    if (!rad)
    { // If the given angles are in degrees, convert them to radians.
        angle_x *= M_PI / 180;
        angle_y *= M_PI / 180;
        fov *= M_PI / 180;
    }

    // x_angle is the angle of the ray around the x-axis.
    // y_angle is the angle of the ray around the y-axis.
    /*    y
        < | >   Λ
    ------ ------ x
          |     V
    */
    // It may be confusing because the x_angle move through the y-axis,
    // and the y_angle move through the x-axis as shown in the diagram.

    Py_ssize_t width = dst_buffer.shape[0];  // width of the screen
    Py_ssize_t height = dst_buffer.shape[1];  // height of the screen

    long *buf = (long *) dst_buffer.buf;  // buffer to write the result in

    // compute a bunch of variables before the loop to avoid computing them at each iteration.

    float cos_x = cosf(angle_x);
    float cos_y = cosf(angle_y);
    float sin_x = sinf(angle_x);
    float sin_y = sinf(angle_y);

    float projection_plane_width = 2 * tanf(fov / 2);
    float projection_plane_height = projection_plane_width * (float) height / (float) width;

    /// top_right(x, y, z) is the position in space of the top right corner of the projection plane (let's say the center of the projection plane is at the pos (0,0,0))

    vec3 top_right = {
            -sin_y * projection_plane_width / 2 - cos_y * sin_x * projection_plane_height / 2,
            projection_plane_height / 2 * cos_x,
            cos_y * projection_plane_width / 2 - sin_y * sin_x * projection_plane_height / 2
    };

    /// forward(x, y, z) is the position in space in front of the camera (let's say the camera is at the pos (0,0,0))

    vec3 forward = {
            cos_y * cos_x,
            sin_x,
            sin_y * cos_x
    };

    vec3 projection = vec3_add(forward, top_right);

    // vector from the top right corner to the top left corner divided by the width of the screen
    vec3 width_vector = {
            sin_y * projection_plane_width / (float) width,
            0,
            -cos_y * projection_plane_width / (float) width
    };

    // vector from the top right corner to the bottom right corner divided by the height of the screen
    vec3 height_vector = {
            cos_y * sin_x * projection_plane_height / (float) height,
            -cos_x * projection_plane_height / (float) height,
            sin_y * sin_x * projection_plane_height / (float) height
    };

    // SHARED DATA between threads
    // all shared data have a t_ prefix
    t_raycaster = self;
    t_view_distance = view_distance;
    t_width = width;
    t_width_vector = width_vector;
    t_A = pos;

    thread_quit = false;
    auto *threads = new std::thread[thread_count];
    for (int i = thread_count - 1; i >= 0; --i)
        threads[i] = std::thread(thread_worker);

    for (Py_ssize_t dst_y = height; dst_y; --dst_y)
    {
        projection = vec3_add(projection, height_vector);

        const struct thread_args t_args{
                .buf = reinterpret_cast<unsigned long *>(reinterpret_cast<unsigned char *>(buf) - 2),
                .pixel_index = dst_y,
                .proj = projection,
        };

        queue_mutex.lock(); // mandatory lock
        args_queue.push(t_args);
        queue_mutex.unlock();

        buf += width;
    }

    thread_quit = true;

    for (int i = thread_count - 1; i >= 0; --i)
    {
        threads[i].join();
    }

    delete[] threads;

    PyBuffer_Release(&dst_buffer);

    free_temp_surfaces(self->surfaces);

    Py_RETURN_NONE;
}

/// Compute the distance to the nearest surface in the given direction
/// If no intersection is found, return the maximum distance.
/// \param ray              starting point and direction
/// \param max_distance     maximum distance to search for
/// \param surfaces         list of surfaces to search in
/// \return           distance to the nearest surface
float get_closest_intersection(const pos2 &ray, float max_distance, const std::vector<struct Surface> &surfaces)
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

/// Compute a single raycast and return the distance from the ray origin to the closest intersection.
/// If no intersection is found, return the max_distance.
/// \param self     the raycaster object
/// \param args     the position arguments
/// \param kwargs   the keyword arguments
/// \return         the distance to the closest intersection. The exact position can be computed by multiplying the returned distance by the direction vector.
static PyObject *method_single_cast(RayCasterObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject * py_origin;

    float angle_x = 0.f;
    float angle_y = 0.f;

    float max_distance = 1000.f;

    bool rad = false;

    static char *kwlist[] = {"origin", "angle_x", "angle_y", "max_distance", "rad", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|fffp", kwlist,
                                     &py_origin, &angle_x, &angle_y, &max_distance, &rad))
        return nullptr;

    vec3 origin;
    if (_get_vec3_from_tuple(py_origin, origin))
        return nullptr;

    if (max_distance <= 0.f)
    {
        PyErr_SetString(PyExc_ValueError, "max_distance must be greater than 0");
        return nullptr;
    }

    if (!rad)
    { // If the given angles are in degrees, convert them to radians.
        angle_x = angle_x * (float) M_PI / 180.f;
        angle_y = angle_y * (float) M_PI / 180.f;
    }

    vec3 destination = {
            cosf(angle_y) * max_distance * cosf(angle_x),
            sinf(angle_x) * max_distance,
            sinf(angle_y) * max_distance * cosf(angle_x)
    };

    struct pos2 ray{
            .A=origin,
            .B=destination,
    };

    return Py_BuildValue("f", get_closest_intersection(ray, max_distance, self->surfaces));
}

/// New method to allocate the Raycaster object
static RayCasterObject *RayCaster_new(PyTypeObject * type)
{
    auto *self = (RayCasterObject *) type->tp_alloc(type, 0);
    if (self == nullptr)
        return nullptr;

    return self;
}


/// Destructor of the raycaster object.
/// \param self    the raycaster object
void RayCaster_dealloc(RayCasterObject *self)
{
    for (struct Surface &surface: self->surfaces)
    {
        free_surface(surface);
    }

    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMethodDef CasterMethods[] = {
        {"add_triangle",   (PyCFunction) method_add_triangle,   METH_VARARGS | METH_KEYWORDS,
                "Adds a triangle to the caster.\n\n"
                ":param image: The pygame surface to display in the scene.\n"
                ":type image: pygame.Surface\n"
                ":param A: The position in space (x,y,z) of the first vertex.\n"
                ":type A: tuple\n"
                ":param B: The position in space (x,y,z) of the second vertex.\n"
                ":type B: tuple\n"
                ":param C: The position in space (x,y,z) of the third vertex.\n"
                ":type C: tuple\n"
                ":param alpha: The alpha value of the surface (1.0 by default).\n"
                ":type alpha: float\n"
                ":param rm: Whether or not the surface should be remove from the scene after being displayed (False by default).\n"
                ":type rm: bool\n"
                ":param reverse: Whether or not the surface should be displayed in reverse (False by default).\n"
                ":type reverse: bool\n"
                ":raise ValueError: If the given image is not a valid surface.\n"},

        {"add_surface",    (PyCFunction) method_add_surface,    METH_VARARGS | METH_KEYWORDS,
                "Adds a quadrilateral (diamond shaped) to the caster.\n\n"
                ":param image: The pygame surface to display in the scene.\n"
                ":type image: pygame.Surface\n"
                ":param A: The position (x,y,z) of the first vertex.\n"
                ":type A: tuple\n"
                ":param B: The position (x,y,z) of the second vertex.\n"
                ":type B: tuple\n"
                ":param C: The position (x,y,z) of the third vertex.\n"
                ":type C: tuple\n"
                ":param alpha: The alpha value of the surface (1.0 by default).\n"
                ":type alpha: float\n"
                ":param rm: Whether or not the surface should be remove from the scene after being displayed (False by default).\n"
                ":type rm: bool\n"
                ":raise ValueError: If the given image is not a valid surface.\n"},

        {"add_quad",       (PyCFunction) method_add_quad,       METH_VARARGS | METH_KEYWORDS,
                "Adds a quad to the scene.\n\n"
                ":param image: The pygame surface to display in the scene.\n"
                ":type image: pygame.Surface\n"
                ":param A: The position in space (x,y,z) of the first vertex.\n"
                ":type A: tuple\n"
                ":param B: The position in space (x,y,z) of the second vertex.\n"
                ":type B: tuple\n"
                ":param C: The position in space (x,y,z) of the third vertex.\n"
                ":type C: tuple\n"
                ":param D: The position in space (x,y,z) of the fourth vertex.\n"
                ":type D: tuple\n"
                ":param alpha: The alpha value of the surface (1.0 by default).\n"
                ":type alpha: float\n"
                ":param rm: Whether or not the surface should be remove from the scene after being displayed (False by default).\n"
                ":type rm: bool\n"
                "raise ValueError: If the given image is not a valid surface.\n"},


        {"add_wall",       (PyCFunction) method_add_wall,       METH_VARARGS | METH_KEYWORDS,
                "Adds a wall to the caster.\n\n"
                ":param image: The pygame surface to display in the scene.\n"
                ":type image: pygame.Surface\n"
                ":param A: The position (x,y,z) of the first vertex.\n"
                ":type A: tuple\n"
                ":param B_x: The position (x,y,z) of the second vertex.\n"
                ":type B_x: tuple\n"
                ":param alpha: The alpha value of the surface (1.0 by default).\n"
                ":type alpha: float\n"
                ":param rm: Whether or not the surface should be remove from the scene after being displayed (False by default).\n"
                ":type rm: bool\n"
                ":raise ValueError: If the given image is not a valid surface.\n"},

        {"clear_surfaces", (PyCFunction) method_clear_surfaces, METH_NOARGS, "Clears all surfaces from the caster."},

        {"add_light",      (PyCFunction) method_add_light,      METH_VARARGS | METH_KEYWORDS,
                "Adds a light to the scene.\n\n"
                ":param position: The position (x,y,z) of the light.\n"
                ":type position: tuple\n"
                ":param intensity: The intensity of the light (1.0 by default).\n"
                ":type intensity: float\n"
                ":param red: The red component of the light (1.0 by default).\n"
                ":type red: float\n"
                ":param green: The green component of the light (1.0 by default).\n"
                ":type green: float\n"
                ":param blue: The blue component of the light (1.0 by default).\n"
                ":type blue: float\n"
                ":param direction: The light direction (x,y,z) (None by default).\n"
                ":type direction: tuple\n"},

        {"clear_lights",   (PyCFunction) method_clear_lights,   METH_NOARGS, "Clears all lights from the caster."},

        {"render",         (PyCFunction) method_raycasting,     METH_VARARGS | METH_KEYWORDS,
                "Render the scene from the given position.\n\n"
                ":param dst_surface: The destination surface.\n"
                ":type dst_surface: pygame.Surface\n"
                ":param pos: The position of the camera.\n"
                ":type pos: tuple\n"
                ":param angle_x: The angle of the camera around the x axis (look up and down).\n"
                ":type angle_x: float\n"
                ":param angle_y: The y angle of the camera around the y axis (look left and right).\n"
                ":type angle_y: float\n"
                ":param fov: The field of view of the camera.\n"
                ":type fov: float\n"
                ":param view_distance: The view distance of the camera.\n"
                ":type view_distance: float\n"
                ":param rad: Whether or not the angles are in radians (False by default).\n"
                ":type rad: bool\n"
                ":param threads: The number of threads to use for the rendering (1 by default). Set to -1 to use the maximum amount of threads.\n"
                ":type threads: int\n"
                ":raise ValueError: If the given destination surface is not a valid surface.\n"
        },

        {"single_cast",    (PyCFunction) method_single_cast,    METH_VARARGS | METH_KEYWORDS,
                "Compute a single raycast and return the distance to the closest intersection.\n\n"
                ":param origin: The position (x,y,z) of the origin of the raycast.\n"
                ":type origin: tuple\n"
                ":param angle_x: The x angle of the ray around the x axis.\n"
                ":type angle_x: float\n"
                ":param angle_y: The y angle of the ray around the y axis.\n"
                ":type angle_y: float\n"
                ":param max_distance: The maximum length of the ray.\n"
                ":type max_distance: float\n"
                ":param rad: Whether or not the angles are in radians (False by default).\n"
                ":type rad: bool\n"
                ":return: The distance to the closest intersection. If no intersection is found, return the max_distance\n"
                ":rtype: float\n"
        },

        {nullptr,          nullptr, 0,                                       nullptr}
};

static PyTypeObject RayCasterType = {
        .ob_base = PyVarObject_HEAD_INIT(nullptr, 0)
        .tp_name = "pysidocast.Scene",
        .tp_basicsize = sizeof(RayCasterObject),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor) RayCaster_dealloc,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = PyDoc_STR("Scene Object"),
        .tp_methods = CasterMethods,
        .tp_new = (newfunc) RayCaster_new,
};


static struct PyModuleDef caster_module = {
        PyModuleDef_HEAD_INIT,
        "pysidocast",
        "Python Ray Caster module for pygame",
        -1,
};


/// Initialize the module.
PyMODINIT_FUNC PyInit_pysidocast(void)
{
    if (PyType_Ready(&RayCasterType) < 0)
        return nullptr;

    PyObject * m = PyModule_Create(&caster_module);

    if (m == nullptr)
        return nullptr;

    Py_INCREF(&RayCasterType);
    if (PyModule_AddObject(m, "Scene", (PyObject *) &RayCasterType) < 0)
    {
        Py_DECREF(&RayCasterType);
        Py_DECREF(m);
        return nullptr;
    }

    return m;
}