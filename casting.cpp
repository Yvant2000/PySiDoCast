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


#define DITHERING_SIZE 16  // Values above 16 will make hardly any difference


/// PyObject containing the RayCaster
typedef struct t_RayCasterObject {
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

    if (*surfaces == nullptr)
        return;

    // The temporary surfaces and the other surfaces are shuffled in the list

    // ensure that the first surface is not temporary
    while((*surfaces) -> del) {
        struct Surface *next = (*surfaces) -> next;
        free_surface(*surfaces);
        if (!(*surfaces = next)) // if all surfaces were temporary, return
            return;
    }

    // free the rest of the surfaces
    struct Surface *prev = *surfaces;
    struct Surface *next;
    for (struct Surface *current = prev -> next; current != nullptr; current = next) {
        next = current -> next;
        if (current->del) {
            free_surface(current);
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
/// \attention For this to work, you must use `.convert_alpha()` on the surface before passing it to this function
/// \param img          pygame surface
/// \param buffer       the buffer from the image
/// \return         true on error, false on success
inline bool _get_3DBuffer_from_Surface(PyObject *img, Py_buffer *buffer) {
    PyObject * get_view_method = PyObject_GetAttrString(img, "get_view");
    if (get_view_method == NULL) {
        printf("Error: Could not get the get_view method from the surface\n");
        return true;
    }

    PyObject *arg = Py_BuildValue("y", "3");
    PyObject *view = PyObject_CallOneArg(get_view_method, arg); // array of width * height * RGBA

    Py_DECREF(arg);
    Py_DECREF(get_view_method);

    if (PyObject_GetBuffer(view, buffer, PyBUF_STRIDES) == -1) {
        Py_DECREF(view);
        printf("Error: Could not get the buffer from the view\n");
        return true;
    }

    Py_DECREF(view);

    return false;
}


/// Gets a float from a tuple
/// \param tuple    the tuple
/// \param index    the index of the item
/// \param result   pointer where the result will be stored
/// \return         0 on success, -1 on error
inline int _get_float_from_tuple(PyObject *tuple, int index, float *result) {
    PyObject *arg = PyLong_FromLong(index);
    PyObject *item;
    if (!(item = PyObject_GetItem(tuple, arg))) {
        printf("Can't access index %d\n", index);
        Py_DECREF(arg);
        return -1;
    }

    *result = (float)PyFloat_AsDouble(item);
    Py_DECREF(arg);
    Py_DECREF(item);

    if (PyErr_Occurred()) {
        printf("Error: Could not convert item %d to float", index);
        return -1;
    }
    return 0;
}

/// Gets a vec3 from a tuple
/// \param tuple    the tuple
/// \param v        pointer where the result will be stored
/// \return         0 on success, -1 on error
inline int _get_vec3_from_tuple(PyObject *tuple, vec3 *v) {
    if (_get_float_from_tuple(tuple, 0, &(v->x))
    || _get_float_from_tuple(tuple, 1, &(v->y))
    || _get_float_from_tuple(tuple, 2, &(v->z)))
        return -1;
    return 0;
}

/// Add a triangle to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_triangle(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *surface_image;

    PyObject *py_A;
    PyObject *py_B;
    PyObject *py_C;

    float alpha = 1.0f;

    bool del = false;
    bool reverse = false;

    static char *kwlist[] = {"image", "A", "B","C","alpha", "rm", "reverse", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|fpp", kwlist,
                                     &surface_image, &py_A, &py_B, &py_C, &alpha, &del, &reverse))
        return NULL;

    vec3 A;
    vec3 B;
    vec3 C;

    if (_get_vec3_from_tuple(py_A, &A)
    || _get_vec3_from_tuple(py_B, &B)
    || _get_vec3_from_tuple(py_C, &C)) {
        return NULL;
    }

    struct Surface *surface = (struct Surface *) malloc(sizeof(struct Surface));
    surface->pos.A = A;
    surface->pos.B = B;
    surface->pos.C = C;

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

/// Add a surface (two triangles) to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_surface(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *surface_image;

    PyObject *py_A;
    PyObject *py_B;
    PyObject *py_C;

    float alpha = 1.0f;

    bool del = false;

    static char *kwlist[] = {"image", "A", "B","C","alpha", "rm", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|fp", kwlist,
                                     &surface_image, &py_A, &py_B, &py_C, &alpha, &del))
        return NULL;

    vec3 A;
    vec3 B;
    vec3 C;

    if (_get_vec3_from_tuple(py_A, &A)
        || _get_vec3_from_tuple(py_B, &B)
        || _get_vec3_from_tuple(py_C, &C)) {
        return NULL;
    }

    struct Surface *surface = (struct Surface *) malloc(sizeof(struct Surface));
    surface->pos.A = A;
    surface->pos.B = B;
    surface->pos.C = C;
    surface->alpha = alpha;
    surface->parent = surface_image;
    surface->del = del;
    surface->reverse = false;

    struct Surface *surface2 = (struct Surface *) malloc(sizeof(struct Surface));
    surface2->pos.A = vec3_sub(vec3_add(C, B), A);
    surface2->pos.B = C;
    surface2->pos.C = B;
    surface2->alpha = alpha;
    surface2->parent = surface_image;
    surface2->del = del;
    surface2->reverse = true;

    if (_get_3DBuffer_from_Surface(surface_image, &surface->buffer)
        || _get_3DBuffer_from_Surface(surface_image, &surface2->buffer)) {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        free(surface);
        free(surface2);
        return NULL;
    }

    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.
    Py_INCREF(surface_image); // Two surfaces means we need to incref twice

    surface->next = surface2;
    surface2->next = self->surfaces; // Push the surface on top of the stack.
    self->surfaces = surface;

    Py_RETURN_NONE;
}




/// Add a quad (two triangles) to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_quad(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *surface_image;

    PyObject *py_A;
    PyObject *py_B;
    PyObject *py_C;
    PyObject *py_D;

    float alpha = 1.0f;

    bool del = false;

    static char *kwlist[] = {"image", "A", "B","C","D","alpha", "rm", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOO|fp", kwlist,
                                     &surface_image, &py_A, &py_B, &py_C, &py_D, &alpha, &del))
        return NULL;

    vec3 A;
    vec3 B;
    vec3 C;
    vec3 D;

    if (_get_vec3_from_tuple(py_A, &A)
        || _get_vec3_from_tuple(py_B, &B)
        || _get_vec3_from_tuple(py_C, &C)
        || _get_vec3_from_tuple(py_D, &D)) {
        return NULL;
    }

    struct Surface *surface = (struct Surface *) malloc(sizeof(struct Surface));
    surface->pos.A = A;
    surface->pos.B = B;
    surface->pos.C = D;
    surface->alpha = alpha;
    surface->parent = surface_image;
    surface->del = del;
    surface->reverse = false;

    struct Surface *surface2 = (struct Surface *) malloc(sizeof(struct Surface));
    surface2->pos.A = C;
    surface2->pos.B = D;
    surface2->pos.C = B;
    surface2->alpha = alpha;
    surface2->parent = surface_image;
    surface2->del = del;
    surface2->reverse = true;

    if (_get_3DBuffer_from_Surface(surface_image, &surface->buffer)
        || _get_3DBuffer_from_Surface(surface_image, &surface2->buffer)) {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        free(surface);
        free(surface2);
        return NULL;
    }

    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.
    Py_INCREF(surface_image); // Two surfaces means we need to incref twice

    surface->next = surface2;
    surface2->next = self->surfaces; // Push the surface on top of the stack.
    self->surfaces = surface;

    Py_RETURN_NONE;
}


/// Add a wall (two triangles) to the list of surfaces in the raycaster
/// \param self The raycaster object
/// \param args The position arguments passed to the function
/// \param kwargs The keyword arguments passed to the function
/// \return (Python) None
static PyObject *method_add_wall(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *surface_image;

    PyObject *py_A;
    PyObject *py_B;

    float alpha = 1.0f;

    bool del = false;

    static char *kwlist[] = {"image", "A", "B","alpha", "rm", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|fp", kwlist,
                                     &surface_image, &py_A, &py_B, &alpha, &del))
        return NULL;

    vec3 A;
    vec3 B;

    if (_get_vec3_from_tuple(py_A, &A)
        || _get_vec3_from_tuple(py_B, &B)) {
        return NULL;
    }

    struct Surface *surface = (struct Surface *) malloc(sizeof(struct Surface));
    surface->pos.A = A;
    surface->pos.B.x = B.x;
    surface->pos.B.y = A.y;
    surface->pos.B.z = B.z;
    surface->pos.C.x = A.x;
    surface->pos.C.y = B.y;
    surface->pos.C.z = A.z;
    surface->alpha = alpha;
    surface->parent = surface_image;
    surface->del = del;
    surface->reverse = false;

    struct Surface *surface2 = (struct Surface *) malloc(sizeof(struct Surface));
    surface2->pos.A = B;
    surface2->pos.B.x = A.x;
    surface2->pos.B.y = B.y;
    surface2->pos.B.z = A.z;
    surface2->pos.C.x = B.x;
    surface2->pos.C.y = A.y;
    surface2->pos.C.z = B.z;
    surface2->alpha = alpha;
    surface2->parent = surface_image;
    surface2->del = del;
    surface2->reverse = true;

    if (_get_3DBuffer_from_Surface(surface_image, &surface->buffer)
        || _get_3DBuffer_from_Surface(surface_image, &surface2->buffer)) {
        PyErr_SetString(PyExc_ValueError, "Not a valid surface");
        free(surface);
        free(surface2);
        return NULL;
    }

    Py_INCREF(surface_image); // We need to keep the surface alive to make sure the buffer is valid.
    Py_INCREF(surface_image); // Two surfaces means we need to incref twice

    surface->next = surface2;
    surface2->next = self->surfaces; // Push the surface on top of the stack.
    self->surfaces = surface;

    Py_RETURN_NONE;
}


static PyObject *method_add_light(RayCasterObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *py_light;

    float light_intensity = 1.0;

    float red = 1.0;  // white by default
    float green = 1.0;
    float blue = 1.0;

    PyObject *py_direction = NULL;

    static char *kwlist[] = {"position", "intensity", "red", "green", "blue", "direction", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ffffO", kwlist,
                                     &py_light, &light_intensity, &red, &green, &blue, &py_direction))
        return NULL;

    vec3 light_pos;
    if (_get_vec3_from_tuple(py_light, &light_pos)) {
        return NULL;
    }

    vec3 light_dir = {FP_NAN, FP_NAN, FP_NAN};
    if (py_direction)
        if (_get_vec3_from_tuple(py_direction, &light_dir))
            return NULL;



    if (red > 1.0f)  // clamp the color values
        red = 1.0f;
    if (green > 1.0f)
        green = 1.0f;
    if (blue > 1.0f)
        blue = 1.0f;

//    // test if one of the direction value is set but not the other
//    if (isnan(direction_x) != isnan(direction_y) || isnan(direction_x) != isnan(direction_z)) {
//        PyErr_SetString(PyExc_ValueError, "Directional light must have all direction values set or none");
//        return NULL;
//    }

    struct Light *light = (struct Light *) malloc(sizeof(struct Light));
    light->pos = light_pos;
    light->intensity = light_intensity;
    light->r = red;
    light->g = green;
    light->b = blue;
    light->direction = light_dir;
    light->next = self->lights;

    self->use_lighting = true;
    self->lights = light;

    self->lights->pos_direction_distance = FP_NAN;
    if (py_direction)
        self->lights->pos_direction_distance = vec3_dist(light_pos, light_dir);

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
    self->surfaces = nullptr;
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

/// Compute weather or not a pixel should be drawn depending of the alpha value
/// \param alpha            alpha value of the pixel
/// \param dither_matrix    dither matrix to use
/// \param x                x coordinate of the pixel on the screen
/// \param y                y coordinate of the pixel on the screen
/// \return         true if the pixel should be skipped, false otherwise
inline bool alpha_dither(float alpha, float *dither_matrix, Py_ssize_t x, Py_ssize_t y) {
    if (!alpha)  // pixel isn't seen at all
        return true;

    if (alpha == 1.0f)  // pixel isn't transparent
        return false;

    return alpha < dither_matrix[(y % DITHERING_SIZE) * DITHERING_SIZE + (x % DITHERING_SIZE)];
}


/// Compute the intersection between a ray and the surfaces in the raycaster and return the color of the pixel
/// \param surfaces surfaces in the scene
/// \param ray ray to cast
/// \param view_distance render distance
/// \return pixel color
inline long get_pixel_at(RayCasterObject *raycaster, struct pos2 ray, Py_ssize_t pixel_index_x, Py_ssize_t pixel_index_y, float view_distance) {
    float closest = view_distance;
    unsigned char *closest_pixel_ptr = nullptr;

    for (struct Surface *surfaces = raycaster->surfaces; surfaces != nullptr; surfaces = surfaces->next) {

        if (alpha_dither(surfaces -> alpha, raycaster->dither_matrix, pixel_index_x, pixel_index_y))
            continue;

        float dist, u, v;
        if (!segment_triangle_intersect(ray, surfaces->pos, closest, &dist, &u, &v))
            continue;

        if (surfaces->reverse) {  // reverse the texture if needed
            u = 1.0f - u;
            v = 1.0f - v;
        }

        unsigned char *new_pixel_ptr = get_pixel_from_buffer(&surfaces->buffer, u, v);
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


mutex queue_mutex;  // Allows only one thread to access the queue at a time
queue<struct thread_args*> args_queue;  // Queue of data that is NOT shared between threads
bool thread_quit;   // Tells the threads to quit once they are done with their current task

// SHARED DATA
// (not so pretty, but it works)
// I store here all the data and shit that will be shared across all the threads

RayCasterObject *t_raycaster;  // Raycaster object
float t_view_distance;  // View distance of the current scene
Py_ssize_t t_width;  // width of the current screen (for some reason we don't need the height)

vec3 t_width_vector;  // vector from the top right corner of the screen to the top left corner divided by the width of the screen
struct vec3 t_A;    // This is the position of the camera


/// data for each individual thread
struct thread_args {          // a few args the thread needs to compute the pixel
    unsigned long *buf;      // where to write the pixel
    Py_ssize_t pixel_index;  // index of the pixel
    vec3 proj;               // projection of the pixel
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

        unsigned long *buf = args -> buf;
        Py_ssize_t pixel_index_y = args -> pixel_index;
        vec3 proj = args -> proj;
        free(args);

        struct pos2 ray;
        ray.A = t_A;
        ray.B = proj;

        for (Py_ssize_t dst_x = t_width; dst_x; --dst_x) {
            long pixel = get_pixel_at(t_raycaster, ray, dst_x, pixel_index_y, t_view_distance);

            if (pixel)
                *buf = pixel;

            ray.B = vec3_add(ray.B, t_width_vector);

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

    PyObject *py_pos;

    float angle_x = 0.f;
    float angle_y = 0.f;

    float fov = 70.f;
    float view_distance = 1000.f;
    bool rad = false;

    int thread_count = 1;

    static char *kwlist[] = {"dst_surface", "pos", "angle_x", "angle_y", "fov", "view_distance", "rad", "threads", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ffffpi", kwlist,
                                     &screen, &py_pos, &angle_x, &angle_y, &fov, &view_distance, &rad, &thread_count))
        return NULL;

    vec3 pos;
    if (_get_vec3_from_tuple(py_pos, &pos))
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

    float cos_x = cosf(angle_x);
    float cos_y = cosf(angle_y);
    float sin_x = sinf(angle_x);
    float sin_y = sinf(angle_y);

    float projection_plane_width = 2 * tanf(fov/2);
    float projection_plane_height = projection_plane_width * (float)height / (float)width;

    /// top_right(x, y, z) is the position in space of the top right corner of the projection plane (let's say the center of the projection plane is at the pos (0,0,0))

    vec3 top_right = {
            -sin_y * projection_plane_width/2 - cos_y * sin_x * projection_plane_height/2,
            projection_plane_height/2 * cos_x,
            cos_y * projection_plane_width/2 - sin_y * sin_x * projection_plane_height/2
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
            sin_y * projection_plane_width / (float)width,
            0,
            -cos_y * projection_plane_width / (float)width
    };

    // vector from the top right corner to the bottom right corner divided by the height of the screen
    vec3 height_vector = {
            cos_y * sin_x * projection_plane_height / (float)height,
            -cos_x * projection_plane_height / (float)height,
            sin_y * sin_x * projection_plane_height / (float)height
    };

    // SHARED DATA between threads
    // all shared data have a t_ prefix
    t_raycaster = self;
    t_view_distance = view_distance;
    t_width = width;
    t_width_vector = width_vector;
    t_A = pos;

    thread_quit = false;
    thread **threads = (thread **)malloc(sizeof(thread *) * thread_count);
    for (int i = thread_count - 1; i >= 0; --i)
        threads[i] = new thread(thread_worker);  // C++ threads 💀 (pthreads ? never heard of them.)

    for (Py_ssize_t dst_y = height; dst_y; --dst_y) {

        projection = vec3_add(projection, height_vector);

        struct thread_args *t_args = (struct thread_args *)malloc(sizeof(struct thread_args));

        t_args -> buf =  (unsigned long *) ((unsigned char *) (buf) - 2);
        t_args -> pixel_index = dst_y;
        t_args -> proj = projection;

        queue_mutex.lock(); // mandatory lock
        args_queue.push(t_args);
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

/// Compute the distance to the nearest surface in the given direction
/// If no intersection is found, return the maximum distance.
/// \param ray              starting point and direction
/// \param max_distance     maximum distance to search for
/// \param surfaces         list of surfaces to search in
/// \return           distance to the nearest surface
float get_closest_intersection(pos2 ray, float max_distance, struct Surface *surfaces) {
    float closest = max_distance;

    for (; surfaces != nullptr; surfaces = surfaces->next) {

        float dist, u, v;
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

    PyObject *py_origin;

    float angle_x = 0.f;
    float angle_y = 0.f;

    float max_distance = 1000.f;

    bool rad = false;

    static char *kwlist[] = {"origin", "angle_x", "angle_y", "max_distance", "rad", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|fffp", kwlist,
                                     &py_origin, &angle_x, &angle_y, &max_distance, &rad))
        return NULL;

    vec3 origin;
    if (_get_vec3_from_tuple(py_origin, &origin))
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
    ray.A = origin;
    ray.B = {cosf(angle_y) * max_distance * cosf(angle_x),
             sinf(angle_x) * max_distance,
             sinf(angle_y) * max_distance * cosf(angle_x)};

    return Py_BuildValue("f", get_closest_intersection(ray, max_distance, self->surfaces));
}


unsigned char bit_reverse(unsigned char n) {
    static unsigned char reverse_lookup[16] = {
            0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
            0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf, };
    // Reverse the top and bottom nibble then swap them.
    return (reverse_lookup[n&0b1111] << 4) | reverse_lookup[n>>4];
}

unsigned int bit_interleave(unsigned char a, unsigned char b) {
    unsigned int z = 0; // z gets the resulting Morton Number.

    for (int i = 0; i < sizeof(a) * CHAR_BIT; i++)
        z |= (a & 1U << i) << i | (b & 1U << i) << (i + 1);

    return z;
}


/// Generate the dither matrix for the given size.
/// \param size     height/width of the matrix
/// \return         pointer to the matrix. The caller is responsible for freeing the memory.
float * generate_dither_matrix(unsigned int size) {
    float *matrix = (float *)malloc(sizeof(float) * size * size);

    for (unsigned int i = 0; i < size; ++i)
        for (unsigned int j = 0; j < size; ++j)
            matrix[i * size + j] = ((float)bit_reverse(bit_interleave(i ^ j, i))) / 256;

    return matrix;
}


/// New method to allocate the Raycaster object
static RayCasterObject *RayCaster_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RayCasterObject *self = (RayCasterObject *) type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    self -> dither_matrix = generate_dither_matrix(DITHERING_SIZE);

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
    struct Light *next_light;
    for (struct Light *light = self->lights; light != nullptr; light = next_light) {
        next_light = light->next;
        free(light);
    }

    free(self->dither_matrix);

    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMethodDef CasterMethods[] = {
        {"add_triangle", (PyCFunction) method_add_triangle, METH_VARARGS | METH_KEYWORDS,
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

        {"add_surface", (PyCFunction) method_add_surface, METH_VARARGS | METH_KEYWORDS,
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

        {"add_quad", (PyCFunction) method_add_quad, METH_VARARGS | METH_KEYWORDS,
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


        {"add_wall", (PyCFunction) method_add_wall, METH_VARARGS | METH_KEYWORDS,
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

        {"add_light", (PyCFunction) method_add_light, METH_VARARGS | METH_KEYWORDS,
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

        {"clear_lights", (PyCFunction) method_clear_lights, METH_NOARGS, "Clears all lights from the caster."},

        {"render", (PyCFunction) method_raycasting, METH_VARARGS | METH_KEYWORDS,
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

        {"single_cast", (PyCFunction) method_single_cast, METH_VARARGS | METH_KEYWORDS,
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

        {NULL, NULL, 0, NULL}
};

static PyTypeObject RayCasterType = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "pysidocast.Scene",
        .tp_basicsize = sizeof(RayCasterObject),
        .tp_itemsize = 0,
        .tp_dealloc = (destructor) RayCaster_dealloc,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_doc = PyDoc_STR("Scene Object"),
        .tp_methods = CasterMethods,
        .tp_new = (newfunc)RayCaster_new,
};


static struct PyModuleDef castermodule = {
    PyModuleDef_HEAD_INIT,
    "pysidocast",
    "Python Ray Caster module for pygame",
    -1,
};


/// Initialize the module.
PyMODINIT_FUNC PyInit_pysidocast(void) {
    if (PyType_Ready(&RayCasterType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&castermodule);

    if (m == NULL)
        return NULL;

    Py_INCREF(&RayCasterType);
    if (PyModule_AddObject(m, "Scene", (PyObject *)&RayCasterType) < 0) {
        Py_DECREF(&RayCasterType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}