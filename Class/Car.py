import pygame
import math

ASSET = pygame.image.load("assets/car.png")
WIDTH = 50
HEIGHT = 25

MAX_SPEED = 300
MAX_REVERSE_SPEED = -100
ACCELERATION = 10
DECELERATION = 10
TURN_SPEED = 4
MIN_TURN_SPEED_THRESHOLD = 30
FRICTION = 0.97


class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.turn = 0
        self.image = pygame.transform.scale(ASSET, (WIDTH, HEIGHT))

    def turn_left(self):
        # if abs(self.speed) > MIN_TURN_SPEED_THRESHOLD:
        self.turn = -TURN_SPEED

    def turn_right(self):
        # if abs(self.speed) > MIN_TURN_SPEED_THRESHOLD:
        self.turn = TURN_SPEED

    def accelerate(self):
        self.speed = min(self.speed + ACCELERATION, MAX_SPEED)

    def decelerate(self):
        self.speed = max(self.speed - DECELERATION, MAX_REVERSE_SPEED)

    def get_mask(self):
        # return hitbox of car
        rotated_image = pygame.transform.rotate(self.image, -self.angle * 180 / math.pi)
        return pygame.mask.from_surface(rotated_image)

    def update(self, dt, level):
        prev_x = self.x
        prev_y = self.y
        prev_angle = self.angle

        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt
        self.angle += self.turn * dt
        self.speed *= FRICTION

        if level.collide(self):
            self.speed = 0
            self.x = prev_x
            self.y = prev_y
            self.angle = prev_angle

        self.turn = 0

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, -self.angle * 180 / math.pi)
        rect = rotated_image.get_rect()
        screen.blit(rotated_image, (self.x - rect.width / 2, self.y - rect.height / 2))

        # bullet


"""         pos = pygame.mouse.get_pos()
        bullet = pygame.Surface((10, 10)) """


"""         # check if bullet collide with car
        if self.get_mask().overlap(
            pygame.mask.from_surface(bullet),
            (pos[0] - (self.x - rect.width / 2), pos[1] - (self.y - rect.height / 2)),
        ):
            bullet.fill((0, 255, 0))
        else:
            bullet.fill((255, 0, 0))

        # draw mask img with good
        screen.blit(
            self.get_mask().to_surface(),
            (self.x - rect.width / 2, self.y - rect.height / 2),
        )

        screen.blit(bullet, (pos[0], pos[1])) """
