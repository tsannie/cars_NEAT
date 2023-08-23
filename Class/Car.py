import pygame
import math

ASSET = pygame.image.load("assets/car.png")
WIDTH = 100
HEIGHT = 50

MAX_SPEED = 300
MAX_REVERSE_SPEED = -100
ACCELERATION = 10
DECELERATION = 10
TURN_SPEED = 0.05
MIN_TURN_SPEED_THRESHOLD = 30
FRICTION = 0.97


class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.image = pygame.transform.scale(ASSET, (WIDTH, HEIGHT))

    def turn_left(self):
        # if abs(self.speed) > MIN_TURN_SPEED_THRESHOLD:
        self.angle += TURN_SPEED

    def turn_right(self):
        # if abs(self.speed) > MIN_TURN_SPEED_THRESHOLD:
        self.angle -= TURN_SPEED

    def accelerate(self):
        self.speed = min(self.speed + ACCELERATION, MAX_SPEED)

    def decelerate(self):
        self.speed = max(self.speed - DECELERATION, MAX_REVERSE_SPEED)

    def update(self, dt):
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt
        self.speed *= FRICTION

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, -self.angle * 180 / math.pi)
        rect = rotated_image.get_rect()
        screen.blit(rotated_image, (self.x - rect.width / 2, self.y - rect.height / 2))
