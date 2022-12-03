#include <Python.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>

#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
    #include <windows.h>
#elif defined(__apple__) || defined(__MACH__)
    #error "not implemented for Mac OS X"
#else  // Linux
    #include <pthread.h>
    #warning "Not implemented for this platform yet"
#endif

//#define EPSILON 0.001f

#define ALPHA 3
#define RED 2
#define GREEN 1
#define BLUE 0

//#define P_RED 3
//#define P_GREEN 2
//#define P_BLUE 1
//#define P_ALPHA 0

//#define MIN(a, b) ((a) < (b) ? (a) : (b))
//#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct t_RayCasterObject{
    PyObject_HEAD
    struct Surface *surfaces = nullptr;
    struct Light *lights = nullptr;
    bool use_lighting = false;
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
        ------ A ------ B ->
    */
    vec3 A;
    vec3 B;
};

struct pos3 {
    /* For a rectangle, the two opposite corners and the normal
        A ------ *
        |  C ->  |
        * ------ B
    */
    /*
        For a triangle, the three corners
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


struct Light {
    // TODO
};


struct Surface {
    struct Surface *next; // linked list
    PyObject *parent;  // The python object that owns this surface
    struct pos3 pos;  // The position of the 3 points of the triangle
    Py_buffer buffer; // The buffer of the surface
    bool del;  // If the Surface is temporary
    bool reverse;  // If the surface texture is reversed (useful for rectangles)
};

inline void free_surface(struct Surface *surface) {
    PyBuffer_Release(&surface->buffer);
//    free(surface->buffer);
    Py_DECREF(surface->parent);
    free(surface);
}

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


inline vec3 vec3_add(vec3 a, vec3 b) {
    vec3 result = {a.x + b.x, a.y + b.y, a.z + b.z};
    return result;
}

/*
 * A - B
 */
inline vec3 vec3_sub(vec3 a, vec3 b) {
    vec3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

inline float vec3_dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline vec3 vec3_dot_float(vec3 a, float b) {
    vec3 result = {a.x * b, a.y * b, a.z * b};
    return result;
}

inline vec3 vec3_cross(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline float vec3_length(vec3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline float vec3_dist(vec3 dot1, vec3 dot2) {
    return sqrtf(powf(dot1.x - dot2.x, 2) + powf(dot1.y - dot2.y, 2) + powf(dot1.z - dot2.z, 2));
}

inline void get_norm_of_plane(vec3 A, vec3 B, vec3 C, vec3 *norm) {

    vec3 AC = vec3_sub(A, C);
    vec3 BC = vec3_sub(B, C);

    *norm = vec3_cross(AC, BC);
}






inline bool _get_3DBuffer_from_Surface(PyObject *img, Py_buffer *buffer) {
    PyObject * get_view_method = PyObject_GetAttrString(img, "get_view");
    if (get_view_method == NULL) {
        return true;
    }

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

    bool del = false;
    bool reverse = false;

    static char *kwlist[] = {"image", "A_x", "A_y", "A_z", "B_x", "B_y", "B_z","C_x", "C_y", "C_z", "rm", "reverse", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Offfffffff|pp", kwlist,
                                     &surface_image, &A_x, &A_y, &A_z, &B_x, &B_y, &B_z, &C_x, &C_y, &C_z, &del, &reverse))
        return NULL;

    struct Surface *surface = (Surface *) malloc(sizeof(struct Surface));
    surface->pos.A.x = A_x;
    surface->pos.A.y = A_y;
    surface->pos.A.z = A_z;
    surface->pos.B.x = B_x;
    surface->pos.B.y = B_y;
    surface->pos.B.z = B_z;
    surface->pos.C.x = C_x;
    surface->pos.C.y = C_y;
    surface->pos.C.z = C_z;

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

    // TODO
    Py_RETURN_NONE;
}

static PyObject *method_clear_surfaces(RayCasterObject *self) {
    struct Surface *next;
    for (struct Surface *surface = self->surfaces; surface != nullptr; surface = next) {
        next = surface->next;
        free_surface(surface);
    }
    Py_RETURN_NONE;
}

static PyObject *method_clear_lights(RayCasterObject *self) {
    // TODO
    Py_RETURN_NONE;
}

/*
 * When the function returns true, the intersection point is given by R.Origin + t * R.Dir.
 * The barycentric coordinates of the intersection in the triangle are u, v, 1-u-v (useful for Gouraud shading or texture mapping)
 */
inline bool segment_triangle_intersect(pos2 segment, pos3 triangle, float closest, float *dist, float *u, float *v) {
    // TODO try to optimize this

    vec3 E1 = vec3_sub(triangle.B, triangle.A);
    vec3 E2 = vec3_sub(triangle.C, triangle.A);
//    vec3 N = vec3_cross(E1, E2);

//    vec3 N = {E1.y * E2.z - E1.z * E2.y,
//              E1.z * E2.x - E1.x * E2.z,
//              E1.x * E2.y - E1.y * E2.x};

    // For some reason the above code is slower than the below code.
    vec3 N = {(triangle.B.y - triangle.A.y) * (triangle.C.z - triangle.A.z) - (triangle.B.z - triangle.A.z) * (triangle.C.y - triangle.A.y),
              (triangle.B.z - triangle.A.z) * (triangle.C.x - triangle.A.x) - (triangle.B.x - triangle.A.x) * (triangle.C.z - triangle.A.z),
              (triangle.B.x - triangle.A.x) * (triangle.C.y - triangle.A.y) - (triangle.B.y - triangle.A.y) * (triangle.C.x - triangle.A.x)};


    float det = -vec3_dot(segment.B, N);
//    float det = -(segment.B.x * ((triangle.B.y - triangle.A.y) * (triangle.C.z - triangle.A.z) - (triangle.B.z - triangle.A.z) * (triangle.C.y - triangle.A.y))
//            + segment.B.y * ((triangle.B.z - triangle.A.z) * (triangle.C.x - triangle.A.x) - (triangle.B.x - triangle.A.x) * (triangle.C.z - triangle.A.z))
//            + segment.B.z * ((triangle.B.x - triangle.A.x) * (triangle.C.y - triangle.A.y) - (triangle.B.y - triangle.A.y) * (triangle.C.x - triangle.A.x)));


//    float inv_det = 1.0f / (
//            segment.B.x * (triangle.B.y - triangle.A.y) * (triangle.C.z - triangle.A.z) - segment.B.x * (triangle.B.z - triangle.A.z) * (triangle.C.y - triangle.A.y)
//            + segment.B.y * (triangle.B.z - triangle.A.z) * (triangle.C.x - triangle.A.x) - segment.B.y * (triangle.B.x - triangle.A.x) * (triangle.C.z - triangle.A.z)
//            + segment.B.z * (triangle.B.x - triangle.A.x) * (triangle.C.y - triangle.A.y) - segment.B.z * (triangle.B.y - triangle.A.y) * (triangle.C.x - triangle.A.x)
//            );

    vec3 AO = vec3_sub(segment.A, triangle.A);
    *dist = vec3_dot(AO, N) / det;
//    *dist = ((segment.A.x - triangle.A.x) * ((triangle.B.y - triangle.A.y) * (triangle.C.z - triangle.A.z) - (triangle.B.z - triangle.A.z) * (triangle.C.y - triangle.A.y))
//            + (segment.A.y - triangle.A.y) * ((triangle.B.z - triangle.A.z) * (triangle.C.x - triangle.A.x) - (triangle.B.x - triangle.A.x) * (triangle.C.z - triangle.A.z))
//            + (segment.A.z - triangle.A.z) * ((triangle.B.x - triangle.A.x) * (triangle.C.y - triangle.A.y) - (triangle.B.y - triangle.A.y) * (triangle.C.x - triangle.A.x))
//            ) / det;

    if (*dist < 0 || *dist >= closest)
        return false;

    vec3 DAO = vec3_cross(AO, segment.B);
//    vec3 DAO = {AO.y * segment.B.z - AO.z * segment.B.y,
//                AO.z * segment.B.x - AO.x * segment.B.z,
//                AO.x * segment.B.y - AO.y * segment.B.x};

    *u = vec3_dot(E2, DAO) / det;
    if (*u < 0)
        return false;

    *v = -vec3_dot(E1, DAO) / det;

    return (*v >= 0. && (*u + *v) <= 1.0);  // -vec3_dot(dir, N) >= EPSILON &&
}

inline unsigned char *get_pixel_from_buffer(Py_buffer *buffer, float u, float v) {
    Py_ssize_t width = buffer -> shape[0];
    Py_ssize_t height = buffer -> shape[1];

    Py_ssize_t x = (Py_ssize_t) (u * width);
    Py_ssize_t y = (Py_ssize_t) (v * height);

    long *buf = (long *)buffer -> buf;
    long *pixel = buf + (y * width + x);
    return ((unsigned char*) pixel) - 2;
}


inline long get_pixel_at(struct Surface *surfaces, struct pos2 ray, float view_distance) {
    float closest = view_distance;
    long pixel = 0;
    unsigned char *closest_pixel_ptr = nullptr;
    float dist;
    float u;
    float v;

    for (; surfaces != nullptr; surfaces = surfaces->next) {
        if (!segment_triangle_intersect(ray, surfaces->pos, closest, &dist, &u, &v))
            continue;

        if (surfaces->reverse) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        unsigned char *new_pixel_ptr = get_pixel_from_buffer(&surfaces->buffer, u, v);
        if (new_pixel_ptr[ALPHA]) {
            closest = dist;
            closest_pixel_ptr = new_pixel_ptr;
        }
    }

    if (closest_pixel_ptr == nullptr)
        return 0;

    float ratio = 1.0f - (closest / view_distance);
    unsigned char *pixel_ptr = (unsigned char *)&pixel;
    pixel_ptr[BLUE] = (unsigned char)(closest_pixel_ptr[BLUE] * ratio);
    pixel_ptr[GREEN] = (unsigned char)(closest_pixel_ptr[GREEN] * ratio);
    pixel_ptr[RED] = (unsigned char)(closest_pixel_ptr[RED] * ratio);
    // pixel_ptr[ALPHA] = new_pixel_ptr[ALPHA];

    return pixel;
}

DWORD WINAPI thread_worker( LPVOID lpParam )
{
    srand(time(NULL));

    void **args = (void **)lpParam;

    unsigned long *buf = (unsigned long *)args[0];

    Surface *surfaces = (Surface *)args[1];

    pos2 *ray_ptr = (pos2 *)args[2];
    pos2 ray = *ray_ptr;
    free(ray_ptr);

    float view_distance = *(float *) args[3];

    free(args);
//
//    printf("THREAD ARGS: \n"
//    "    buf = %p\n"
//    "    surfs = %p\n"
//    "    view_distance = %f\n"
//    "    A - %f %f %f\n"
//    "    B - %f %f %f\n", buf, surfaces, view_distance, ray.A.x, ray.A.y, ray.A.z, ray.B.x, ray.B.y, ray.B.z);

    long pixel = get_pixel_at(surfaces, ray, view_distance);
    if (pixel)
        *buf = pixel;

    ExitThread(0);
}


/*
 * Get the index of a free thread in the thread pool.
 * If no free thread is available, wait for one to become free.
 */
inline void wait_thread(HANDLE thread) {
    if (thread == nullptr)
        return;

    DWORD exit_code;
    GetExitCodeThread(thread, &exit_code);
    if (exit_code != STILL_ACTIVE)
        return;

    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
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

    if (thread_count < 1) {
        PyErr_SetString(PyExc_ValueError, "thread_count must be greater than 0");
        return NULL;
    }

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

    // x_angle is the angle of the ray around the x axis.
    // y_angle is the angle of the ray around the y axis.
    /*    y
        < | >   Λ
    ------ ------ x
          |     V
    */
    // It may be confusing because the x_angle move through the y axis,
    // and the y_angle move through the x axis as shown in the diagram.

//    struct pos2 ray;
//    ray.A = {x, y, z};

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
    float d_progress_x = 1.f / (float)width;

    float progress_y = 0.5f;


    HANDLE *threads = (HANDLE *)malloc(sizeof(HANDLE) * thread_count);
    int thread_index = 0;
    for (int i = thread_count; i; --i)
        threads[i] = nullptr;  // init threads to nullptr

//    omp_set_num_threads(thread_count);
    pos2 ray;
    ray.A = {x, y, z};

    for (Py_ssize_t dst_y = height; dst_y; --dst_y) {

        progress_y -= d_progress_y;
//        progress_y = 0.5 - d_progress_y * dst_y;

        // float y_ = forward_y + progress_y * right_y;
        ray.B.y = forward_y + progress_y * right_y;

        float progress_x = 0.5f;

//        #pragma omp parallel for
        for (Py_ssize_t dst_x = width; dst_x; --dst_x) {
            wait_thread(threads[thread_index]); // Ensure that the thread is free before using it.

            progress_x -= d_progress_x;
            // float progress_x = 0.5f - dst_x * d_progress_x;

            ray.B.x = forward_x + progress_x * right_x;
            // ray.B.y = y_;
            ray.B.z = forward_z + progress_x * right_z;
//
//            long pixel = get_pixel_at(self->surfaces, ray, view_distance);
//
//            if (pixel)   // If the pixel is empty, don't draw it.
//                *((unsigned long *) ((unsigned char *) (buf) - 2)) = pixel;
//                // *((unsigned long *) ((unsigned char *) (buf + dst_y * width + dst_x) - 2)) = pixel;

            pos2 *ray_copy = (pos2 *)malloc(sizeof(pos2));
            *ray_copy = ray;

//            printf("EXPECTED ARGS: \n"
//                   "    buf = %p\n"
//                   "    surfs = %p\n"
//                   "    view_distance = %f\n"
//                   "    A - %f %f %f\n"
//                   "    B - %f %f %f\n", (unsigned long *) ((unsigned char *) (buf) - 2), self->surfaces, view_distance, ray.A.x, ray.A.y, ray.A.z, ray.B.x, ray.B.y, ray.B.z);

            void **args = (void **)malloc(sizeof(void *) * 4);
            args[0] = (unsigned long *) ((unsigned char *) (buf) - 2);
            args[1] = self->surfaces;
            args[2] = ray_copy;
            args[3] = &view_distance;

            threads[thread_index] = CreateThread(NULL, 0, thread_worker, args, 0, NULL);

            thread_index = (thread_index + 1) % thread_count;

            buf++;
        }

    }

    WaitForMultipleObjects(thread_count, threads, TRUE, INFINITE);
//    for (int i = thread_count; i; --i)
//        CloseHandle(threads[i]);

    PyBuffer_Release(&dst_buffer);

    free_temp_surfaces(&(self->surfaces));

    free(threads);

    Py_RETURN_NONE;
}


/*
 *  compute a single raycast and return the position in space of the closest intersection.
 */
static PyObject *method_single_cast(RayCasterObject *self, PyObject *args, PyObject *kwargs) {

    // TODO
    Py_RETURN_NONE;
}


void RayCaster_dealloc(RayCasterObject *self) {
    struct Surface *next;
    for (struct Surface *surface = self->surfaces; surface != nullptr; surface = next) {
        next = surface->next;
        free_surface(surface);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
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
        .tp_new = PyType_GenericNew,
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
