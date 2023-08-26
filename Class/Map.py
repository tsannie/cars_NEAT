import pygame

from Class.Flag import Flag

COLOR = (240, 0, 240)
COLOR_FILL = (10, 10, 10)


class Map:
    def __init__(self, size_screen):
        self.surface = pygame.Surface(size_screen)
        self.surface.fill(COLOR_FILL)

        self.start = None
        self.end = None

    def setFlag(self, pos, type_flag):
        if type_flag == "start":
            self.start = Flag(pos[0], pos[1], type_flag)
        elif type_flag == "end":
            self.end = Flag(pos[0], pos[1], type_flag)
        self.add_circle(pos[0], pos[1], 40)

    def add_circle(self, x, y, radius):
        pygame.draw.circle(self.surface, COLOR, (x, y), radius)

    def ready(self):
        return self.start and self.end

    def collide(self, car):
        mask_map = self.get_mask()
        mask_car = car.get_mask()

        if mask_map.overlap(
            mask_car,
            (
                car.x - mask_car.get_rect().width / 2,
                car.y - mask_car.get_rect().height / 2,
            ),
        ):
            return True
        return False

    def get_mask(self):
        return pygame.mask.from_threshold(self.surface, COLOR_FILL, (1, 1, 1, 255))

    def draw(self, screen):
        # mask = self.get_mask()
        # screen.blit(mask.to_surface(), (0, 0))
        screen.blit(self.surface, (0, 0))
        if self.start:
            self.start.draw(screen)
        if self.end:
            self.end.draw(screen)

        # draw self.get_mask().to_surface() with good
        # mask = self.get_mask()
        # change color to red
        # screen.blit(mask.to_surface(), (0, 0))

    # screen.blit(self.get_mask().to_surface(), (0, 0))
