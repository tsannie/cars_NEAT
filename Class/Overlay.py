import pygame

ALL_RADIUS = {
    0: 30,
    1: 50,
    2: 70,
}
CIRCLE_SPACING = 20
DEFAULT_RADIUS = 1


class Overlay:
    def __init__(self, size_screen):
        self.selection_radius = DEFAULT_RADIUS
        self.size_screen = size_screen

    def draw_overlay(self, screen):
        last = None
        for i in range(3):
            pass
