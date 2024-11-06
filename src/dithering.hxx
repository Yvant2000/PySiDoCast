#ifndef CASTING_CPP_DITHERING_HXX
#define CASTING_CPP_DITHERING_HXX

#include <array>
#include <Python.h>

static constexpr const int DITHERING_SIZE = 16;  // Values above 16 will make hardly any difference

static consteval unsigned int bit_interleave(unsigned char a, unsigned char b)
{
    unsigned int z = 0; // z gets the resulting Morton Number.

    for (int i = 0; i < sizeof(a) * CHAR_BIT; i++)
        z |= (a & 1U << i) << i | (b & 1U << i) << (i + 1);

    return z;
}

static consteval unsigned char bit_reverse(unsigned char n)
{
    constexpr std::array<char, 16> reverse_lookup = {
            0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf
    };

    // Reverse the top and bottom nibble then swap them.
    return (reverse_lookup[n & 0b1111] << 4) | reverse_lookup[n >> 4];
}

/// Generate the dither matrix for the given size.
/// \param size     height/width of the matrix
/// \return         pointer to the matrix. The caller is responsible for freeing the memory.
static consteval std::array<float, DITHERING_SIZE * DITHERING_SIZE> generate_dither_matrix()
{
    std::array<float, DITHERING_SIZE * DITHERING_SIZE> matrix{};

    for (unsigned int i = 0; i < DITHERING_SIZE; ++i)
        for (unsigned int j = 0; j < DITHERING_SIZE; ++j)
            matrix[i * DITHERING_SIZE + j] = ((float) bit_reverse(bit_interleave(i ^ j, i))) / 256;

    return matrix;
}

static constinit const std::array<float, DITHERING_SIZE * DITHERING_SIZE> DITHER_MATRIX = generate_dither_matrix();

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

#endif //CASTING_CPP_DITHERING_HXX
