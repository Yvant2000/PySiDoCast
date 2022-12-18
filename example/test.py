

import pysidocast
import pygame
from math import *

from os.path import join as join_path

pygame.init()

clock = pygame.time.Clock()

caster = pysidocast.RayCaster()
caster2 = pysidocast.RayCaster()

infoObject = pygame.display.Info()

mult = 0.4
dim = (infoObject.current_w * mult, infoObject.current_h * mult)
print(dim)
unit = 1
view_distance = 3 * unit
screen = pygame.display.set_mode(dim, pygame.SCALED | pygame.FULLSCREEN)
game_screen = pygame.Surface(dim)
image = pygame.image.load(".\\example\\bg.png").convert_alpha()
transp = pygame.image.load(".\\example\\transp.png").convert_alpha()
transp2 = pygame.image.load(".\\example\\transp2.png").convert_alpha()
transp3 = pygame.image.load(".\\example\\transp3.png").convert_alpha()

caster.add_surface(transp,
                   unit, 2 * unit, 2 * unit,
                   unit, 2 * unit, 0,
                   unit, 0, 2 * unit)

caster.add_surface(transp,
                   unit, 0, 0,
                   unit, 0, 2 * unit,
                   unit, 2*unit, 0,
                   reverse=True)

caster.add_surface(image,
                   1.5*unit, 2*unit, 2.5*unit,
                   1.5*unit, 2*unit, .5*unit,
                   1.5*unit, 0, 2.5*unit)

caster.add_surface(image,
                   1.5*unit, 0, .5*unit,
                   1.5*unit, 0, 2.5*unit,
                   1.5*unit, 2*unit, .5*unit,
                   reverse=True)

caster.add_surface(transp2,
                   -unit, 2*unit, 0,
                   -unit, 2*unit, 2*unit,
                   -unit, 0, 0)

caster.add_surface(transp2,
                   -unit, 0, 2*unit,
                   -unit, 0, 0,
                   -unit, 2*unit, 2*unit,
                   reverse=True)

caster.add_surface(image,
                   -unit+0.5, 2*unit, 2*unit,
                   unit, 2*unit, 2*unit,
                   -unit, 0, 2*unit)

caster.add_surface(image,
                   unit-0.5, 0, 2*unit,
                   -unit, 0, 2*unit,
                   unit, 2*unit, 2*unit,
                   reverse=True)


caster.add_surface(image,
                   -unit, 0.0, 2*unit,
                   unit, 0.0, 2*unit,
                   -unit, 0.0, 0)

caster.add_surface(image,
                   unit, 0.0, 0,
                   -unit, 0.0, 0,
                   unit, 0.0, 2*unit,
                   reverse=True)

caster.add_surface(image,
                   -2*unit, 2*unit, unit,
                   -2*unit, 2*unit, 0,
                   -3*unit, 0, unit)

caster.add_surface(image,
                   -3*unit, 0, 0,
                   -3*unit, 0, unit,
                   -2*unit, 2*unit, 0,
                   reverse=True)


spot = 0
alpha = 0.0
if __name__ == "__main__":
    y_angle = 90
    x_angle = 0
    x = 0
    z = -0.5*unit
    y = unit
    time = 0
    speed = unit/3
    while True:

        keys = pygame.key.get_pressed()
        time_stamp = clock.tick(60) / 100
        fps = clock.get_fps()
        print(fps)
        game_screen.fill((0, 0, 0, 0))
        # print(fps, time_stamp)
        # if fps < 30:
        #     mult *= 0.95
        #     del game_screen
        #     game_screen = pygame.Surface((int(dim[0] * mult), int(dim[1] * mult)))
        # elif fps > 40:
        #     if mult <= 0.5:
        #         mult *= 1.05
        #         del game_screen
        #         game_screen = pygame.Surface((int(dim[0] * mult), int(dim[1] * mult)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # put the mouse in the center of the screen if the window is active
        if pygame.key.get_focused():
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            # get the mouse position relative to the window's center
            rel = pygame.mouse.get_rel()
            if rel[0] or rel[1]:
                pygame.mouse.set_pos(dim[0] // 2, dim[1] // 2)
            x_angle -= (rel[1] / 10)
            y_angle -= (rel[0] / 10)
            x_angle = max(min(x_angle, 90), -90)
        else:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

        if keys[pygame.K_z]:
            z += time_stamp * speed * sin(radians(y_angle))
            x += time_stamp * speed * cos(radians(y_angle))

        elif keys[pygame.K_s]:
            z -= time_stamp * speed * sin(radians(y_angle))
            x -= time_stamp * speed * cos(radians(y_angle))

        if keys[pygame.K_q]:
            z += time_stamp * speed * sin(radians(y_angle + 90))
            x += time_stamp * speed * cos(radians(y_angle + 90))

        elif keys[pygame.K_d]:
            z -= time_stamp * speed * sin(radians(y_angle + 90))
            x -= time_stamp * speed * cos(radians(y_angle + 90))

        if keys[pygame.K_LEFT]:
            y_angle += time_stamp * 15
        elif keys[pygame.K_RIGHT]:
            y_angle -= time_stamp * 15

        if keys[pygame.K_UP]:
            x_angle += time_stamp * 15
        elif keys[pygame.K_DOWN]:
            x_angle -= time_stamp * 15

        if keys[pygame.K_SPACE]:
            y += time_stamp * speed
        elif keys[pygame.K_LSHIFT]:
            y -= time_stamp * speed

        caster.clear_lights()

        caster.add_light(
                         x, y, z,
                         view_distance, 0.5, 0.6, 0.7,
                         direction_x=x + cos(radians(y_angle)) * view_distance * 1.8,
                         direction_y=y + sin(radians(x_angle)) * view_distance * 1.8,
                         direction_z=z + sin(radians(y_angle)) * view_distance * 1.8,
                         )
        caster.add_light(0, 0, unit/2, unit * 2, 0.5, 0.3, 0.2)

        caster.add_light(cos(spot) * unit, unit * 2, unit + sin(spot) * unit,
                         unit*3, 0.3, 0.3, 1.0,
                         direction_x=unit * sin(spot), direction_y=-3*unit, direction_z= unit + cos(spot) * unit)

        spot += 0.07 * time_stamp
        alpha = (alpha + 0.01 * time_stamp) % 1.0

        caster.add_surface(transp3,
                           -unit, 2*unit, 0,
                           unit, 2*unit, 0,
                           -unit, 0, 0,
                           alpha=alpha,
                           rm=True)

        caster.add_surface(transp3,
                           unit, 0, 0,
                           -unit, 0, 0,
                           unit, 2*unit, 0,
                           reverse=True,
                           alpha=alpha,
                           rm=True)



        from timeit import repeat, default_timer as timer

        # start = timer()

        # end = timer()
        # print(end - start)

        test = lambda: caster.raycasting(game_screen,
                                          x, y, z,
                                          x_angle, y_angle,
                                          fov=60,
                                          view_distance=view_distance,
                                          threads=1)

        test()

        # print(repeat(test, repeat=5, number=10))
        # exit()

        pygame.transform.scale(game_screen, dim, screen)

        # screen.blit(game_screen, (0, 0))

        pygame.display.update()
        # y_angle += 1
        # x_angle += 1
        time += 1
