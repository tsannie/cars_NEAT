import pygame


class Map:
    def __init__(self):
        self.surface = pygame.Surface((1280, 720))

    def add_circle(self, x, y, radius):
        pygame.draw.circle(self.surface, (240, 240, 240), (x, y), radius)

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))
