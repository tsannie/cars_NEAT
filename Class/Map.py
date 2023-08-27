import pygame

from Class.Flag import Flag

COLOR = (57, 89, 65)
COLOR_FILL = (155, 191, 188)
COLOR_UNBREAKABLE = (0, 0, 0)


class Map:
    def __init__(self, size_screen):
        self.surface = pygame.Surface(size_screen)
        self.surface.fill(COLOR_FILL)

        # draw the border
        self.border = pygame.Rect(0, 0, size_screen[0], size_screen[1])

        self.start = None
        self.end = None

        self.mask = None

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

    def compute_mask(self):
        # draw the border
        pygame.draw.rect(self.surface, COLOR_FILL, self.border, 1)

        self.mask = self.get_mask()

    def collide(self, car):
        mask_map = self.mask
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

    def collide_point(self, x, y):
        mask_map = self.mask

        if not self.border.collidepoint(x, y):
            return True

        if mask_map.get_at((int(x), int(y))):
            return True
        return False

    def get_mask(self):
        return pygame.mask.from_threshold(self.surface, COLOR_FILL, (1, 1, 1, 255))

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))
        if self.start:
            self.start.draw(screen)
        if self.end:
            self.end.draw(screen)
