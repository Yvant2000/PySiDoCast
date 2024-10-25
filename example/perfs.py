import time

import pysidocast
import pygame
from timeit import default_timer as timer

pygame.display.set_mode((1, 1))  # init pygame

dims = (3840, 2160)
render_image = pygame.surface.Surface(dims)
image = pygame.image.load(".\\example\\bg.png").convert_alpha()  # WARNING: the ".convert_alpha()" is mandatory
scene = pysidocast.Scene()  # INIT THE SCENE

# perf variables
WALLS = 100
LIGHTS = 100
REPEAT_WALLS = 1000
REPEAT_LIGHTS = 10000

print("start performance test")

# test load walls
start = timer()
for _ in range(REPEAT_WALLS):
    scene.clear_surfaces()
    for _ in range(WALLS):
        scene.add_wall(image, (3, 1, 1), (3, -1, -1))
end = timer()
print(f"add_wall: {end - start}")

# test add lights
start = timer()
for _ in range(REPEAT_LIGHTS):
    scene.clear_lights()
    for _ in range(LIGHTS):
        scene.add_light((3, 0, 0))
end = timer()
print(f"add_light: {end - start}")

# test display walls
start = timer()
scene.render(render_image, (0, 0, 0))
end = timer()
print(f"render {WALLS} walls and {LIGHTS} lights: {end - start}")

"""
add_wall: 0.15190609999990556
add_light: 0.1368188000014925
render 100 walls and 100 lights: 9.697650599999179
"""
