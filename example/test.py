import pysidocast
import pygame
from math import *

pygame.init()

scene = pysidocast.Scene()  # INIT THE SCENE

PERF = 0.32  # Decrease this value to increase the performance (but decrease the quality)
dim = (1280 * PERF, 720 * PERF)

view_distance = 7

mouse_sensitivity = 0.1

screen = pygame.display.set_mode(dim, pygame.SCALED)  # | pygame.FULLSCREEN

clock = pygame.time.Clock()

# LOAD IMAGES

image = pygame.image.load(".\\example\\bg.png").convert_alpha()  # WARNING: the ".convert_alpha()" is mandatory
red_blue_gradiant = pygame.image.load(".\\example\\red_blue.png").convert_alpha()
transparent = pygame.image.load(".\\example\\transparent.png").convert_alpha()
uniform = pygame.image.load(".\\example\\white.png").convert_alpha()

# If your wondering why the ".convert_alpha()" is mandatory,
# it's because in pygame, the pixel buffer will differ depending on the image format.
# Using the ".convert_alpha()" will convert the image to the same format as the screen.
# In future versions, I will try to make the ".convert_alpha()" optional.
# But anyway, there is no reason for you to not use ".convert_alpha()" in pygame, so don't worry about it.


# LOAD STATIC SURFACES

scene.add_wall(red_blue_gradiant,
               (1, 2, 2),
               (1, 0, 0))

scene.add_wall(image,
               (1.5, 2, 0.5),
               (1.5, 0, 2.5))

scene.add_wall(transparent,
               (-1, 2, 0),
               (-1, 0, 2))

scene.add_quad(image,
               (-1 + 0.5, 2, 2),
               (0.75, 1.75, 2),
               (1, 0.5, 2),
               (-1, 0, 2))

scene.add_surface(image,
                  (-1, 0.0, 2),
                  (1, 0.0, 2),
                  (-1, 0.0, 0))

scene.add_surface(image,
                  (-2, 2, 1),
                  (-2, 2, 0),
                  (-3, 0, 1))

spot = 0
alpha = 0.0
if __name__ == "__main__":
    y_angle = 90.  # look toward the Z axis ( 0° = look toward the X axis; 90° = look toward the Z axis)
    x_angle = 0.  # look toward the horizon ( 90° = look toward the sky; -90° = look toward the ground)
    x = 0.
    z = 1.
    y = 1.
    speed = 1. / 3.6  # 1 km/h in m/s

    # MAIN LOOP
    while True:
        # RESET THE SCREEN
        screen.fill((0, 0, 0, 0))

        # HANDLE THE TIME AND THE FPS
        keys = pygame.key.get_pressed()
        time_stamp = clock.tick(60) / 100
        # fps = clock.get_fps()
        # print(fps)

        # HANDLE THE EVENTS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # HANDLE THE INPUTS

        if pygame.key.get_focused():  # Place the mouse in the center of the screen when the window is active
            pygame.mouse.set_visible(False)  # Hide the mouse
            pygame.event.set_grab(True)  # Grab the mouse
            rel = pygame.mouse.get_rel()  # Get the mouse movement since the last frame
            if rel[0] or rel[1]:  # When the mouse move, replace it in the center of the screen
                pygame.mouse.set_pos(dim[0] // 2, dim[1] // 2)
            x_angle -= rel[1] * mouse_sensitivity
            y_angle -= rel[0] * mouse_sensitivity
            x_angle = max(min(x_angle, 90), -90)  # Limit the angle to prevent the player from looking upside down
        else:
            # Give back the mouse control when the window is not active
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

        # Move forward
        if any((keys[pygame.K_z], keys[pygame.K_w], keys[pygame.K_UP])):
            z += time_stamp * speed * sin(radians(y_angle))
            x += time_stamp * speed * cos(radians(y_angle))
        # Move backward
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            z -= time_stamp * speed * sin(radians(y_angle))
            x -= time_stamp * speed * cos(radians(y_angle))
        # Move left
        if any((keys[pygame.K_q], keys[pygame.K_a], keys[pygame.K_LEFT])):
            z += time_stamp * speed * sin(radians(y_angle + 90))
            x += time_stamp * speed * cos(radians(y_angle + 90))
        # Move right
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            z -= time_stamp * speed * sin(radians(y_angle + 90))
            x -= time_stamp * speed * cos(radians(y_angle + 90))

        # Move up
        if keys[pygame.K_SPACE]:
            y += time_stamp * speed
        # Move down
        elif keys[pygame.K_LSHIFT]:
            y -= time_stamp * speed

        #  ___________________RENDER THE SCENE

        # PLACE THE LIGHTS INTO THE SCENE

        scene.clear_lights()  # Clear the lights. You need to do this anytime a light moves or changes.

        # Add a directional light from the player position to where the player is looking
        scene.add_light(
            (x, y, z),  # position of the light
            view_distance,  # strength of the light
            0.5, 0.6, 0.7,  # color of the light
            # direction of the light
            direction=(x + cos(radians(y_angle)) * cos(radians(x_angle)) * view_distance * 1.8,
                       y + sin(radians(x_angle)) * view_distance,
                       z + sin(radians(y_angle)) * cos(radians(x_angle)) * view_distance * 1.8),
        )

        scene.add_light((0, 0, 0.5), 2, 0.3, 0.6, 0.2)  # add a green light at the origin

        # add a directional light that moves around the scene
        scene.add_light((cos(spot), 2, 1 + sin(spot)),
                        3, 0.3, 0.3, 1.0,
                        direction=(sin(spot), -3, cos(spot)))

        spot += 0.07 * time_stamp  # move the light

        # ADD THE DYNAMIC SURFACES INTO THE SCENE

        alpha = (alpha + 0.01 * time_stamp) % 2.0  # Change the alpha value of the surface over time

        # Add a surface that disappears and reappears using the alpha value
        scene.add_wall(uniform,
                       (-1, 2, 0),
                       (1, 0, 0),
                       alpha=abs(alpha - 1.0),  # alpha value goes from 0.0 ⏫ to 1.0 ⏬ to 0.0
                       rm=True)

        # RENDER THE SCENE

        scene.render(screen,
                     (x, y, z),
                     x_angle, y_angle,
                     fov=120,
                     view_distance=view_distance,
                     threads=-1)

        pygame.display.update()
