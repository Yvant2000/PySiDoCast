#ifndef CASTING_CPP_DITHERING_H
#define CASTING_CPP_DITHERING_H

#include <array>
#include <Python.h>

static constexpr const int DITHERING_SIZE = 16;  // Values above 16 will make hardly any difference
extern constinit const std::array<float, DITHERING_SIZE * DITHERING_SIZE> DITHER_MATRIX;

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

#endif //CASTING_CPP_DITHERING_H
