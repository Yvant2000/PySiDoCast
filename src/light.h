#ifndef CASTING_CPP_LIGHT_H
#define CASTING_CPP_LIGHT_H

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

#endif //CASTING_CPP_LIGHT_H
