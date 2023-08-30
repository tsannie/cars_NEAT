import pygame

END_ASSET = pygame.image.load("assets/flag.png")
START_ASSET = pygame.image.load("assets/green.png")
WITDT = 50
HEIGHT = 50


class Flag:
    def __init__(self, x, y, type_flag):
        self.x = x
        self.y = y

        self.mask = None

        if type_flag == "end":
            self.image = pygame.transform.scale(END_ASSET, (WITDT, HEIGHT))
        elif type_flag == "start":
            self.image = pygame.transform.scale(START_ASSET, (WITDT, HEIGHT))
        else:
            raise Exception("type_flag must be 'end' or 'start'")

    def draw(self, screen):
        rect = self.image.get_rect()
        screen.blit(self.image, (self.x - rect.width / 2, self.y - rect.height / 2))

        if self.mask:
            msk = self.mask.outline()

    def compute_mask(self):
        self.mask = pygame.mask.from_surface(self.image)

    def collide(self, car):
        mask_flag = self.mask
        mask_car = car.get_mask()

        offset = (
            int(self.x - car.x),
            int(self.y - car.y),
        )

        if mask_flag.overlap(mask_car, offset):
            return True
        return False
