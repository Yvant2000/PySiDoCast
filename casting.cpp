#include <Python.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <mutex>
#include <thread>
#include <queue>

using namespace std;  // I hate cpp and its namespaces

// #define EPSILON 0.001f

#define ALPHA 3
#define RED 2
#define GREEN 1
#define BLUE 0

/// PyObject containing the RayCaster
typedef struct t_RayCasterObject{
    PyObject_HEAD           // required python object header
    struct Surface *surfaces = nullptr;     // List of surfaces
    struct Light *lights = nullptr;         // List of lights
    float *dither_matrix = nullptr;         // Dither matrix
    bool use_lighting = false;              // Use lighting or not
} RayCasterObject;



typedef struct vec3 {
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


struct pos2 {
    /* For a segment, the start point and end point.
        A [---------] B
    */
    /* For a line, a point and a direction.
        ------ A ------ (B) ->
    */
    vec3 A;
    vec3 B;
};

struct pos3 {
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
struct Light {
    vec3 pos;  // Position of the light in the scene
    vec3 direction;  // Direction of the light in the scene
    float pos_direction_distance;  // Distance between the light and the direction
    float intensity; // Intensity of the light (= the distance of lightning)
    float r; // Red component of the light
    float g; // Green component of the light
    float b; // Blue component of the light
    struct Light *next;  // Next light in the list
    // TODO: Might add an intensity offset (useful for cel shading)
};

///
/// \brief A surface in the scene
struct Surface {
    Py_buffer buffer; // The buffer of the surface
    struct pos3 pos;  // The position of the 3 points of the triangle
    struct Surface *next; // linked list
    PyObject *parent;  // The python object that owns this surface
    // We need to keep a reference to the parent object to prevent the buffer from being destroyed by the garbage collector
    bool del;  // If the Surface is temporary and needs to be deleted
    bool reverse;  // If the surface texture is reversed (useful for rectangles)
    float alpha;  // The alpha value of the surface
};


/// Free a surface object
/// \param surface The surface to free
inline void free_surface(struct Surface *surface) {
    PyBuffer_Release(&surface->buffer);
    Py_DECREF(surface->parent);
    free(surface);
}

/// Free temporary surfaces in the list
/// \param surfaces The list of surfaces
inline void free_temp_surfaces(struct Surface **surfaces) {
    struct Surface *prev = nullptr;
    struct Surface *next;
    for (struct Surface *current = *surfaces; current != nullptr; current = next) {
        next = current->next;
        if (current->del) {
            free_surface(current);
            if (prev == nullptr)
                *surfaces = next;
            else
                prev->next = next;
        } else
            prev = current;
    }
}

/// Sum two vectors
/// \param a
/// \param b
/// \return a + b
inline vec3 vec3_add(vec3 a, vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

/// Subtract two vectors
/// \param a
/// \param b
/// \return a - b
inline vec3 vec3_sub(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

/// Dot product of two vectors
/// \param a : vector(3)
/// \param b : vector(3)
/// \return a . b : scalar
inline float vec3_dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// Multiply a vector by a scalar
/// \param a : vector(3)
/// \param b : scalar(1)
/// \return a * b : vector(3)
inline vec3 vec3_dot_float(vec3 a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

/// Cross product of two vectors
/// \param a
/// \param b
/// \return a x b
inline vec3 vec3_cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

/// Normalize a vector
/// \param a : vector(3)
/// \return |a| : scalar
inline float vec3_length(vec3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

/// Compute distance between two points
/// \param a : vector(3)
/// \param b : vector(3)
/// \return |a - b| : scalar
inline float vec3_dist(vec3 dot1, vec3 dot2) {
    return sqrtf(powf(dot1.x - dot2.x, 2) + powf(dot1.y - dot2.y, 2) + powf(dot1.z - dot2.z, 2));
}

/// gets the py_buffer from a pygame surface
/// \param img pygame surface
/// \param buffer the buffer from the image
/// \return true on error, false on success
inline bool _get_3DBuffer_from_Surface(PyObject *img, Py_buffer *buffer) {
    PyObject * get_view_method = PyObject_GetAttrString(img, "get_view");
    if (get_view_method == NULL)
        return true;

    PyObject *arg = Py_BuildValue("y", "3");
    PyObject * view = PyObject_CallOneArg(get_view_method, arg); // array of width * height * RGBA

    Py_DECREF(arg);
    Py_DECREF(get_view_method);

    if (PyObject_GetBuffer(view, buffer, PyBUF_STRIDES) == -1) {
        Py_DECREF(view);
        return true;
    }

    Py_DECREF(view);

    return false;
}

/// Add a surface to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_surface(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *surface_image;

    float A_x;
    float A_y;
    float A_z;

    float B_x;
    float B_y;
    float B_z;

    float C_x;
    float C_y;
    float C_z;

    float alpha = 1.0f;

    bool del = false;
    bool reverse = false;

    static char *kwlist[] = {"image", "A_x", "A_y", "A_z", "B_x", "B_y", "B_z","C_x", "C_y", "C_z","alpha", "rm", "reverse", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Offfffffff|fpp", kwlist,
                                     &surface_image, &A_x, &A_y, &A_z, &B_x, &B_y, &B_z, &C_x, &C_y, &C_z, &alpha, &del, &reverse))
        return NULL;

    struct Surface *surface = (struct Surface *) malloc(sizeof(struct Surface));
    surface->pos.A.x = A_x;
    surface->pos.A.y = A_y;
    surface->pos.A.z = A_z;
    surface->pos.B.x = B_x;
    surface->pos.B.y = B_y;
    surface->pos.B.z = B_z;
    surface->pos.C.x = C_x;
    surface->pos.C.y = C_y;
    surface->pos.C.z = C_z;

    surface->alpha = alpha;

    surface->parent = surface_image;
    surface->del = del;
    surface->reverse = reverse;

    if (_get_3DBuffer_from_Surface(surface_image, &surface->buffer)) {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        free(surface);
        return NULL;
    }
    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.

    surface->next = self->surfaces; // Push the surface on top of the stack.
    self->surfaces = surface;

    Py_RETURN_NONE;
}


static PyObject *method_add_light(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    float light_x;
    float light_y;
    float light_z;

    float light_intensity = 1.0;

    float red = 1.0;  // white by default
    float green = 1.0;
    float blue = 1.0;

    float direction_x = FP_NAN;  // default value for non directional lights
    float direction_y = FP_NAN;
    float direction_z = FP_NAN;

    static char *kwlist[] = {"x", "y", "z", "intensity", "red", "green", "blue", "direction_x", "direction_y", "direction_z", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "fff|fffffff", kwlist, &light_x, &light_y, &light_z, &light_intensity,
                                     &red, &green, &blue, &direction_x, &direction_y, &direction_z))
        return NULL;

    if (red > 1.0f)  // clamp the color values
        red = 1.0f;
    if (green > 1.0f)
        green = 1.0f;
    if (blue > 1.0f)
        blue = 1.0f;

    struct Light *light = (struct Light *) malloc(sizeof(struct Light));
    light->pos.x = light_x;
    light->pos.y = light_y;
    light->pos.z = light_z;
    light->intensity = light_intensity;
    light->r = red;
    light->g = green;
    light->b = blue;
    light->direction.x = direction_x;
    light->direction.y = direction_y;
    light->direction.z = direction_z;
    light->next = self->lights;

    self->use_lighting = true;
    self->lights = light;

    self->lights->pos_direction_distance = FP_NAN;
    if (direction_x != FP_NAN && direction_y != FP_NAN && direction_z != FP_NAN)
        self->lights->pos_direction_distance = vec3_dist(self->lights->pos, self->lights->direction);

    Py_RETURN_NONE;
}

/// Free all the surfaces in the raycaster
/// \param self  The raycaster object
/// \return (Python) None
static PyObject *method_clear_surfaces(RayCasterObject *self) {
    struct Surface *next;
    for (struct Surface *surface = self->surfaces; surface != nullptr; surface = next) {
        next = surface->next;
        free_surface(surface);
    }
    Py_RETURN_NONE;
}

/// Free all the lights in the raycaster
/// \param self  The raycaster object
/// \return (Python) None
static PyObject *method_clear_lights(RayCasterObject *self) {
    struct Light *next;
    for (struct Light *light = self->lights; light != nullptr; light = next) {
        next = light->next;
        free(light);
    }
    self->lights = nullptr;
    self->use_lighting = false;
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
inline bool segment_triangle_intersect(pos2 segment, pos3 triangle, float closest, float *dist, float *u, float *v) {
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
inline unsigned char *get_pixel_from_buffer(Py_buffer *buffer, float u, float v) {
    Py_ssize_t width = buffer -> shape[0];
    Py_ssize_t height = buffer -> shape[1];

    Py_ssize_t x = (Py_ssize_t) (u * width);
    Py_ssize_t y = (Py_ssize_t) (v * height);

    long *buf = (long *)buffer -> buf;
    long *pixel = buf + (y * width + x);
    return ((unsigned char*) pixel) - 2;
}


/// Compute the minimum distance between a point and a line
/// \param point position of the point
/// \param line_point position of a point on the line
/// \param line_direction vector representing the direction of the line
/// \return distance
inline float line_point_distance(vec3 point, vec3 line_point, vec3 line_direction) {
    vec3 s = vec3_sub(line_direction, line_point);
    vec3 w = vec3_sub(point, line_point);
    float ps = vec3_dot(w, s);

    if (ps <= 0)
        return vec3_length(w);

    float l2 = vec3_dot(s, s);
    if (ps >= l2)
        return vec3_length(vec3_sub(point, line_direction));

    return vec3_length(vec3_sub(point, vec3_add(line_point, vec3_dot_float(s, ps / l2))));
}

inline bool alpha_dither(float alpha, float *dither_matrix, Py_ssize_t x, Py_ssize_t y) {
    if (!alpha)  // pixel isn't seen at all
        return true;

    if (alpha == 1.0f)  // pixel isn't transparent
        return false;

    return alpha > dither_matrix[(y % 32) * 32 + (x % 32)];  // TODO custom dether size
}



///// If a surface is transparent, tells weather or not the current pixel should be ignored
//inline bool alpha_mask(float alpha, Py_ssize_t pixel_index_x, Py_ssize_t pixel_index_y) {
//    if (!alpha)  // pixel isn't seen at all
//        return true;
//
//    if (alpha == 1.0f)  // pixel isn't transparent
//        return false;
//
//    if (alpha <= 0.5) {
//        // less than 50% opaque
//        Py_ssize_t ratio = (Py_ssize_t)(1.f / alpha);
//        if (pixel_index_x % ratio)
//            return true;
//        if (pixel_index_y % ratio)
//            return true;
//
//    } else {
//        // more than 50% opaque
//        Py_ssize_t ratio = (Py_ssize_t)(1.f / (1.f - alpha));
//        if (!(pixel_index_x % ratio))
//            return true;
//        if (!(pixel_index_y % ratio))
//            return true;
//    }
//
//    return false;
//}

/// Compute the intersection between a ray and the surfaces in the raycaster and return the color of the pixel
/// \param surfaces surfaces in the scene
/// \param ray ray to cast
/// \param view_distance render distance
/// \return pixel color
inline long get_pixel_at(RayCasterObject *raycaster, struct pos2 ray, Py_ssize_t pixel_index_x, Py_ssize_t pixel_index_y, float view_distance) {
    float closest = view_distance;
    unsigned char *closest_pixel_ptr = nullptr;

    for (struct Surface *surfaces = raycaster->surfaces; surfaces != nullptr; surfaces = surfaces->next) {

//        if (alpha_mask(surfaces -> alpha, pixel_index_x, pixel_index_y))
//            continue;

        if (alpha_dither(surfaces -> alpha, raycaster->dither_matrix, pixel_index_x, pixel_index_y))
            continue;

        float dist;
        float u;
        float v;
        if (!segment_triangle_intersect(ray, surfaces->pos, closest, &dist, &u, &v))
            continue;

        if (surfaces->reverse) {  // reverse the texture if needed
            u = 1.0f - u;
            v = 1.0f - v;
        }

        unsigned char *new_pixel_ptr = get_pixel_from_buffer(&surfaces->buffer, u, v); // TODO move this out of the loop if possible
//        if (alpha_mask(new_pixel_ptr[ALPHA] / 255.f, pixel_index_x, pixel_index_y))
//            continue;
        if (alpha_dither(new_pixel_ptr[ALPHA] / 255.f, raycaster->dither_matrix, pixel_index_x, pixel_index_y))
            continue;

        closest = dist;
        closest_pixel_ptr = new_pixel_ptr;

    }

    if (closest_pixel_ptr == nullptr)
        return 0;

    long pixel = 0;
    unsigned char *pixel_ptr = (unsigned char *)&pixel;

    float r = 1.0f - (closest / view_distance);
    pixel_ptr[BLUE] = (unsigned char)(closest_pixel_ptr[BLUE] * r);
    pixel_ptr[GREEN] = (unsigned char)(closest_pixel_ptr[GREEN] * r);
    pixel_ptr[RED] = (unsigned char)(closest_pixel_ptr[RED] * r);
    // pixel_ptr[ALPHA] = new_pixel_ptr[ALPHA];

    if (!raycaster->use_lighting || !pixel)
        return pixel;

    // apply lights

    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;

    // position in space of the pixel
    vec3 inter = vec3_add(ray.A, vec3_dot_float(ray.B, closest)); // inter = A + t * B

    for (struct Light* temp_light = raycaster->lights; temp_light != nullptr; temp_light = temp_light->next) {  // iterate over lights
        float dist1 = vec3_dist(temp_light->pos, inter);  // distance between the light and the intersection
        float ratio = temp_light->pos_direction_distance == FP_NAN ?
                // if the light is a radial light, calculate the ratio
                    dist1 / temp_light->intensity :
                // if the light is a directional light, calculate the ratio
                    (line_point_distance(inter, temp_light->pos, temp_light->direction)  // distance between the direction and the intersection
                    * temp_light->pos_direction_distance) / (dist1 * temp_light->intensity);

        if (ratio < 1.0f) {  // ratio > 1 means the light is too far away, we don't see anything
            float temp = 1.0f - ratio;
            red += temp * temp_light->r;
            green += temp * temp_light->g;
            blue += temp * temp_light->b;
        }
    }
    // Prevent the pixel from being too bright
    if (red > 1.0f)
        red = 1.0f;
    if (green > 1.0f)
        green = 1.0f;
    if (blue > 1.0f)
        blue = 1.0f;

    pixel_ptr[BLUE] = (unsigned char)(pixel_ptr[BLUE] * blue);
    pixel_ptr[GREEN] = (unsigned char)(pixel_ptr[GREEN] * green);
    pixel_ptr[RED] = (unsigned char)(pixel_ptr[RED] * red);

    return pixel;
}


// SHARED DATA
// (not so pretty, but it works)
// I store here all the data and shit that will be shared across all the threads

mutex queue_mutex;  // Allows only one thread to access the queue at a time
queue<struct thread_args*> args_queue;  // Queue of data that is NOT shared between threads
bool thread_quit;   // Tells the threads to quit once they are done with their current task
RayCasterObject *t_raycaster;  // Raycaster object
float t_view_distance;  // View distance of the current scene
Py_ssize_t t_width;  // width of the current screen (for some reason we don't need the height)
float t_forward_x;  // I don't remember what this is, but it's used in the ray calculation
float t_forward_z;  // No idea
float t_right_x;    // i forgor 💀
float t_right_z;    // Don't touch this anyway
struct vec3 t_A;    // This is the position of the camera (i rember 😁)


struct thread_args {          // a few args the thread needs to compute the pixel
    unsigned long *buf;      // where to write the pixel
    Py_ssize_t pixel_index;  // index of the pixel
    float y;
};


void thread_worker()
{
    while(true)  // while the job isn't done
    {
        queue_mutex.lock();  // wait until I get the lock
        if (args_queue.empty())  // check if there is job to do
        {
            queue_mutex.unlock(); // if not release the lock
            if (thread_quit)  // if the job is done and we need to quit
                return;

            this_thread::sleep_for(chrono::microseconds (1));  // wait a bit before checking again to not waste cpu
            continue;
        }

        struct thread_args *args = args_queue.front();  // get my args from the queue
        args_queue.pop();
        queue_mutex.unlock();  // release the lock so other threads can get their args


        unsigned long *buf = args->buf;
        Py_ssize_t pixel_index_y = args->pixel_index;
        float y = args->y;
        free(args);

        struct pos2 ray;
        ray.A = t_A;
        ray.B.y = y;


        float progress_x = 0.5f;
        float d_progress_x = 1.0f / (float)t_width;

        for (Py_ssize_t dst_x = t_width; dst_x; --dst_x) {
            progress_x -= d_progress_x;
            // float progress_x = 0.5f - dst_x * d_progress_x;

            ray.B.x = t_forward_x + progress_x * t_right_x;
            // ray.B.y = y_;
            ray.B.z = t_forward_z + progress_x * t_right_z;

            long pixel = get_pixel_at(t_raycaster, ray, dst_x, pixel_index_y, t_view_distance);
            if (pixel)
                *buf = pixel;

//            pixel_index++;
            buf++;
        }


    }
//    printf("THREAD ARGS: \n"
//    "    buf = %p\n"
//    "    surfs = %p\n"
//    "    view_distance = %f\n"
//    "    A - %f %f %f\n"
//    "    B - %f %f %f\n", buf, surfaces, view_distance, ray.A.x, ray.A.y, ray.A.z, ray.B.x, ray.B.y, ray.B.z);

}


static PyObject *method_raycasting(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *screen;

    float x = 0.f;
    float y = 0.f;
    float z = 0.f;

    float angle_x = 0.f;
    float angle_y = 0.f;

    float fov = 120.f;
    float view_distance = 1000.f;
    bool rad = false;

    int thread_count = 1;

    static char *kwlist[] = {"dst_surface", "x", "y", "z", "angle_x", "angle_y", "fov", "view_distance", "rad", "threads", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|fffffffpi", kwlist,
                                     &screen, &x, &y, &z, &angle_x, &angle_y, &fov, &view_distance, &rad, &thread_count))
        return NULL;

    if(fov <= 0.f) {
        PyErr_SetString(PyExc_ValueError, "fov must be greater than 0");
        return NULL;
    }
    if (view_distance <= 0.f) {
        PyErr_SetString(PyExc_ValueError, "view_distance must be greater than 0");
        return NULL;
    }

    if (thread_count == 0) {
        PyErr_SetString(PyExc_ValueError, "thread_count can't be 0");
        return NULL;
    } else if (thread_count < 0)
        thread_count = thread::hardware_concurrency();


    Py_buffer dst_buffer;
    if (_get_3DBuffer_from_Surface(screen, &dst_buffer)) {
        PyErr_SetString(PyExc_ValueError, "dst_surface is not a valid surface");
        return NULL;
    }

    if (!rad) { // If the given angles are in degrees, convert them to radians.
        angle_x = angle_x * (float)M_PI / 180.f;
        angle_y = angle_y * (float)M_PI / 180.f;
        fov = fov * (float)M_PI / 180.f;
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

    long *buf = (long *)dst_buffer.buf;  // buffer to write the result in

    // compute a bunch of variables before the loop to avoid computing them at each iteration.

    float projection_plane_width = 2 * tan(fov);
    float projection_plane_height = projection_plane_width * (float)height / (float)width;

    float forward_x = cosf(angle_y) * view_distance;
    float forward_y = sinf(angle_x) * projection_plane_height * view_distance;
    float forward_z = sinf(angle_y) * view_distance;

    float right_x = -forward_z * projection_plane_width;
    float right_y = projection_plane_height * view_distance;
    float right_z = forward_x * projection_plane_width;

    float d_progress_y = 1.f / (float)height;
//    float d_progress_x = 1.f / (float)width;

//    struct pos2 ray;
//    ray.A = {x, y, z};

    // SHARED DATA between threads
    // all shared data have a t_ prefix
    t_raycaster = self;
    t_view_distance = view_distance;
    t_width = width;
    t_forward_x = forward_x;
    t_forward_z = forward_z;
    t_right_x = right_x;
    t_right_z = right_z;
    t_A = {x, y, z};

    thread_quit = false;
    thread **threads = (thread **)malloc(sizeof(thread *) * thread_count);
    for (int i = thread_count - 1; i >= 0; --i)
        threads[i] = new thread(thread_worker);  // C++ threads 💀 (pthreads ? never heard of them.)


    float progress_y = 0.5f;
    for (Py_ssize_t dst_y = height; dst_y; --dst_y) {

        progress_y -= d_progress_y;
//        progress_y = 0.5 - d_progress_y * dst_y;

        // ray.B.y = forward_y + progress_y * right_y; // computed once for each thread


        struct thread_args *args = (struct thread_args *)malloc(sizeof(struct thread_args));

        args -> buf =  (unsigned long *) ((unsigned char *) (buf) - 2);
        args -> pixel_index = dst_y;
        args -> y = forward_y + progress_y * right_y;

        queue_mutex.lock(); // mandatory lock
        args_queue.push(args);
        queue_mutex.unlock();

        buf += width;
    }

    thread_quit = true;

    for (int i = thread_count - 1; i >= 0; --i) {
        threads[i] -> join();
        delete threads[i];
    }
    free(threads);

    PyBuffer_Release(&dst_buffer);

    free_temp_surfaces(&(self->surfaces));

    Py_RETURN_NONE;
}


float get_closest_intersection(pos2 ray, float max_distance, struct Surface *surfaces) {
    float closest = max_distance;
    float dist;
    float u;
    float v;

    for (; surfaces != nullptr; surfaces = surfaces->next) {

        if (!segment_triangle_intersect(ray, surfaces->pos, closest, &dist, &u, &v))
            continue;

        unsigned char *new_pixel_ptr = get_pixel_from_buffer(&surfaces->buffer, u, v);
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
static PyObject *method_single_cast(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    float origin_x = 0.f;
    float origin_y = 0.f;
    float origin_z = 0.f;

    float angle_x = 0.f;
    float angle_y = 0.f;

    float max_distance = 1000.f;

    bool rad = false;

    static char *kwlist[] = {"x", "y", "z", "angle_x", "angle_y", "max_distance", "rad", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ffffffp", kwlist,
                                     &origin_x, &origin_y, &origin_z, &angle_x, &angle_y, &max_distance, &rad))
        return NULL;

    if (max_distance <= 0.f) {
        PyErr_SetString(PyExc_ValueError, "max_distance must be greater than 0");
        return NULL;
    }

    if (!rad) { // If the given angles are in degrees, convert them to radians.
        angle_x = angle_x * (float)M_PI / 180.f;
        angle_y = angle_y * (float)M_PI / 180.f;
    }

    struct pos2 ray;
    ray.A = {origin_x, origin_y, origin_z};
    ray.B = {cosf(angle_y) * max_distance, sinf(angle_x) * max_distance, sinf(angle_y) * max_distance};

    return Py_BuildValue("f", get_closest_intersection(ray, max_distance, self->surfaces));
}

static unsigned char lookup[16] = {
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
        0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf, };


unsigned char bit_reverse(unsigned char n) {
    // Reverse the top and bottom nibble then swap them.
    return (lookup[n&0b1111] << 4) | lookup[n>>4];
}

unsigned int bit_interleave(unsigned char a, unsigned char b) {
    unsigned int z = 0; // z gets the resulting Morton Number.

    for (int i = 0; i < sizeof(a) * CHAR_BIT; i++) // unroll for more speed...
        z |= (a & 1U << i) << i | (b & 1U << i) << (i + 1);

    return z;
}


float * generate_dither_matrix(unsigned int size) {
    float *matrix = (float *)malloc(sizeof(float) * size * size);

    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            matrix[i * size + j] = ((float)bit_reverse(bit_interleave(i ^ j, i))) / 256;
        }
    }

    return matrix;
}


static RayCasterObject *RayCaster_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RayCasterObject *self = (RayCasterObject *) type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    unsigned int dithering = 32;

    self -> dither_matrix = generate_dither_matrix(dithering);

    return (RayCasterObject *) self;
}


/// Destructor of the raycaster object.
/// \param self    the raycaster object
void RayCaster_dealloc(RayCasterObject *self) {
    struct Surface *next;
    for (struct Surface *surface = self->surfaces; surface != nullptr; surface = next) {
        next = surface->next;
        free_surface(surface);
    }
    free(self->dither_matrix);

    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyMethodDef CasterMethods[] = {
        {"add_surface", (PyCFunction) method_add_surface, METH_VARARGS | METH_KEYWORDS, "Adds a surface to the caster."},
        {"clear_surfaces", (PyCFunction) method_clear_surfaces, METH_NOARGS, "Clears all surfaces from the caster."},
        {"add_light", (PyCFunction) method_add_light, METH_VARARGS | METH_KEYWORDS, "Adds a light to the scene."},
        {"clear_lights", (PyCFunction) method_clear_lights, METH_NOARGS, "Clears all lights from the caster."},
        {"raycasting", (PyCFunction) method_raycasting, METH_VARARGS | METH_KEYWORDS, "Display the scene using raycasting."},
        {"single_cast", (PyCFunction) method_single_cast, METH_VARARGS | METH_KEYWORDS, "Compute a single raycast and return the position in space of the closest intersection."},
        {NULL, NULL, 0, NULL}
};

static PyTypeObject RayCasterType = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "pysidocast.RayCaster",
        .tp_basicsize = sizeof(RayCasterObject),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor) RayCaster_dealloc,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = PyDoc_STR("RayCaster Object"),
        .tp_methods = CasterMethods,
        .tp_new = (newfunc)RayCaster_new,
};


static struct PyModuleDef castermodule = {
    PyModuleDef_HEAD_INIT,
    "pysidocast",
    "Python Ray Caster module for pygame",
    -1,
};


PyMODINIT_FUNC PyInit_pysidocast(void) {
    if (PyType_Ready(&RayCasterType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&castermodule);

    if (m == NULL)
        return NULL;

    Py_INCREF(&RayCasterType);
    if (PyModule_AddObject(m, "RayCaster", (PyObject *)&RayCasterType) < 0) {
        Py_DECREF(&RayCasterType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}