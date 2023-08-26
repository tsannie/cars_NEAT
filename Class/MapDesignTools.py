import pygame

START_ASSET = pygame.image.load("assets/green.png")
END_ASSET = pygame.image.load("assets/flag.png")
SIZE_ASSET = (64, 64)

ALL_RADIUS = {
    0: 15,
    1: 30,
    2: 40,
}
COLOR = (240, 240, 240)
COLOR_SELECTED = (100, 100, 100)
COLOR_ERROR = (236, 46, 46)
DEFAULT_RADIUS = 1

SIZE_BUTTON = (130, 130)


class MapDesignTools:
    def __init__(self, size_screen):
        self.selection = DEFAULT_RADIUS
        self.size_screen = size_screen

    def getRadius(self):
        if self.selection in ALL_RADIUS:
            return ALL_RADIUS[self.selection]
        return None

    def draw_radius_button(self, screen, i, font):
        rect = pygame.Rect(
            SIZE_BUTTON[1] * i,
            self.size_screen[1] - SIZE_BUTTON[1],
            SIZE_BUTTON[0],
            SIZE_BUTTON[1],
        )
        pygame.draw.rect(
            screen, COLOR_SELECTED if i == self.selection else COLOR, rect, 3
        )

        circle = pygame.draw.circle(
            screen,
            COLOR,
            (int(rect.x + rect.width / 2), int(rect.y + rect.height / 2)),
            ALL_RADIUS[i],
        )
        pygame.draw.circle(screen, (0, 0, 0), circle.center, ALL_RADIUS[i], 2)

        text = font.render(str(i + 1), True, COLOR)
        screen.blit(
            text,
            (rect.x + 10, rect.y + 10),
        )

    def draw_image_button(self, screen, i, font, asset):
        rect = pygame.Rect(
            SIZE_BUTTON[1] * i,
            self.size_screen[1] - SIZE_BUTTON[1],
            SIZE_BUTTON[0],
            SIZE_BUTTON[1],
        )
        pygame.draw.rect(
            screen, COLOR_SELECTED if i == self.selection else COLOR, rect, 3
        )

        img = pygame.transform.scale(asset, SIZE_ASSET)
        screen.blit(
            img,
            (
                rect.x + rect.width / 2 - SIZE_ASSET[0] / 2,
                rect.y + rect.height / 2 - SIZE_ASSET[1] / 2,
            ),
        )

        text = font.render(str(i + 1), True, COLOR)
        screen.blit(
            text,
            (rect.x + 10, rect.y + 10),
        )

    def draw_msg(self, screen, msg, font, postion, color):
        text = font.render(msg, True, color)
        screen.blit(text, (10, 0 + 20 * postion))

    def draw(self, screen, startIsSet, endIsSet):
        last = None
        font = pygame.font.SysFont("Arial", 20)

        for i in range(len(ALL_RADIUS)):
            self.draw_radius_button(screen, i, font)

        self.draw_image_button(screen, 3, font, START_ASSET)
        self.draw_image_button(screen, 4, font, END_ASSET)

        if not startIsSet:
            self.draw_msg(screen, "NO START FLAG", font, 1, COLOR_ERROR)
        if not endIsSet:
            self.draw_msg(screen, "NO END FLAG", font, 2, COLOR_ERROR)
        if startIsSet and endIsSet:
            self.draw_msg(screen, "PRESS SPACE TO START", font, 3, COLOR)
