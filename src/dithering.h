#ifndef CASTING_CPP_DITHERING_H
#define CASTING_CPP_DITHERING_H

#include <array>

static constexpr const int DITHERING_SIZE = 16;  // Values above 16 will make hardly any difference
extern constinit const std::array<float, DITHERING_SIZE*DITHERING_SIZE> DITHER_MATRIX;

#endif //CASTING_CPP_DITHERING_H
