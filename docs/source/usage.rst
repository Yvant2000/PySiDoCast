Usage
=====

.. _installation:

Installation
------------

To use PySiDoCast, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pysidocast

Creating a Scene
----------------

To create a new scene, use ``pysidocast.Scene()`` :

.. autoclass:: pysidocast.Scene

For example:

>>> import pysidocast
>>> scene = pysidocast.Scene()


Loading Surfaces into the Scene
-------------------------------

First, let's load a surface using pygame:

>>> import pygame
>>> pygame.init()
>>> screen = pygame.display.set_mode((600, 600))
>>> image = pygame.image.load("image.png").convert_alpha()



You can then add surfaces to the scene.

.. note:: Each surface should be added only once in the scene.

Make sure that this code isn't in the Main Loop.

If for any reason you need to add a surface multiple times
(for example, if you want to change the position of the surface over time),
then set the ``rm`` argument to True to remove the surface after each render.

.. automethod:: pysidocast.Scene.add_triangle(image, (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), alpha=1.0, rm=False, reverse=False)

.. note:: The ``add_triangle()`` method display a triangle into the scene. To display a quadrilateral, you should use the ``add_surface()`` method.

For example:

>>> scene.add_triangle(image, (-1, 2, 1), (1, 2, 1), (-1, 0, 1))
>>> scene.render(screen, (0, 1, 0), 0, 90, 60)
>>> pygame.display.update()

.. image:: https://media.discordapp.net/attachments/914913842260217898/1065373419572564088/image.png
    :alt: A triangle in the scene



.. automethod:: pysidocast.Scene.add_surface(image, (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), alpha=1.0, rm=False)

.. note:: With the ``add_surface()`` method, the fourth vertex of the quadrilateral is automatically calculated to make a diamond shape.


For example:

>>> scene.add_surface(image, (-0.5, 2, 1), (1, 2, 1), (-1, 0, 1))
>>> scene.render(screen, (0, 1, 0), 0, 90, 60)
>>> pygame.display.update()

.. image:: https://cdn.discordapp.com/attachments/914913842260217898/1065376365366485133/image.png
    :alt: A plane in the scene



.. automethod:: pysidocast.Scene.add_quad(image, (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4), alpha=1.0, rm=False)

.. note:: The ``add_quad()`` method is usefull for custom shapes, but most of the time it's easier to use ``add_wall()`` or ``add_surface()``.



.. automethod:: pysidocast.Scene.add_wall(image, (x2,y2,z2), (x2,y2,z2), alpha=1.0, rm=False)

.. note:: With the ``add_wall()`` method, the second and fourth vertex of the quadrilateral is automatically calculated to make a rectangle shape.


For example:

>>> scene.add_wall(image, (-1, 2, 1), (1, 0, 1))
>>> scene.render(screen, (0, 1, 0), 0, 90, 60)
>>> pygame.display.update()

.. image:: https://media.discordapp.net/attachments/914913842260217898/1065377127966449684/image.png
    :alt: A wall in the scene


Removing Surfaces from the Scene
--------------------------------

The surfaces with the ``rm`` argument set to True are removed from the scene after each render.

If you need to remove all surfaces from the scene, use the ``clear_surfaces()`` method.

.. automethod:: pysidocast.Scene.clear_surfaces()

.. note:: This method is automatically called if the scene object is garbage collected.


Adding Lights to the Scene
--------------------------

By default, the lights are disabled, meaning that the scene is fully lit.
To use custom lights, use the ``add_light()`` method.

.. automethod:: pysidocast.Scene.add_light((x, y, z), intensity=1.0, red=1.0, green=1.0, blue=1.0, direction=(None,None,None))

.. note:: By default, the light is a radial light. If you want to use a directional light, set the ``direction`` argument.

Example with radial light:

>>> scene.add_light((0, 1, 1))
>>> scene.add_wall(image, (-1, 2, 1), (1, 0, 1))
>>> scene.render(screen, (0, 1, 0), 0, 90, 60)
>>> pygame.display.update()

.. image:: https://media.discordapp.net/attachments/914913842260217898/1065378419296186418/image.png
    :alt: A wall with a light in the scene

Example with directional light:

>>> scene.add_light((-1, 2, 1), direction=(1, 0, 1))
>>> scene.add_wall(image, (-1, 2, 1), (1, 0, 1))
>>> scene.render(screen, (0, 1, 0), 0, 90, 60)
>>> pygame.display.update()

.. image:: https://media.discordapp.net/attachments/914913842260217898/1065381141844078652/image.png
    :alt: A well with a directionnal light in the scene


To remove all lights from the scene, use the ``clear_lights()`` method.

.. automethod:: pysidocast.Scene.clear_lights()

Calling this method will disable the lights until the ``add_light()`` method is called again.

.. note:: This method is automatically called if the scene object is garbage collected.


Render the Scene
----------------

You can now render the scene using the ``render()`` method.

.. automethod:: pysidocast.Scene.render(dst_surface, (x,y,z), angle_x=0.0, angle_y=0.0, fov=120.0, view_distance=1000.0, rad=False, threads=1)

The scene will be drawn over the destination surface.

.. note: The original surface will not be cleared, therefore you can draw a skybox or a background before rendering the scene.

The ``x``, ``y`` and ``z`` arguments are the position of the camera.

The ``angle_x`` and ``angle_y`` arguments are the rotation of the camera.

The ``fov`` argument is the field of view of the camera.

The ``view_distance`` argument is the maximum distance that the camera can see.

The ``rad`` argument is a boolean that indicates if the angles needs to be converted to radians.
To use degrees, set this argument to False. To use radians, set this argument to True.

Be default, only one thread will be use to render the scene.
If you want to use multiple threads, set the ``threads`` argument to -1 or a positive integer.

.. warning:: This method might not be thread-safe.


You can also cast a single ray in the scene using the ``single_cast()`` method.

This method is useful to check if a point is visible from a specific position.

It will return the distance between the camera and the closest point in a given direction.

If no point is found, it will return the maximum view distance.

.. automethod:: pysidocast.Scene.single_cast((x,y,z), angle_x=0.0, angle_y=0.0, max_distance=1000.0, rad=False)

