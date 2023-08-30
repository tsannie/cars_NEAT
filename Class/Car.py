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
        self.distance = []
        self.finished = False

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
            self.speed = -self.speed * 0.5
            self.x = prev_x
            self.y = prev_y
            self.angle = prev_angle

        self.turn = 0

        if level.end.collide(self):
            self.finished = True

    def get_point_with_distance(self, distance, angle):
        x = self.x + math.cos(angle) * distance
        y = self.y + math.sin(angle) * distance
        return x, y

    def compute_distance_angle(self, level, angle):
        # get the distance between the car and the wall
        distance = 0
        while distance < 2000:
            distance += 1
            x, y = self.get_point_with_distance(distance, angle)
            if level.collide_point(x, y):
                return distance, x, y

    def get_distances(self, level, screen):
        self.distance = []
        for angle in range(-2, 3):
            distance, x, y = self.compute_distance_angle(
                level, self.angle + angle * 0.5
            )
            self.distance.append((distance))
            pygame.draw.line(screen, (0, 0, 170), (self.x, self.y), (x, y), 1)

        return self.distance

    def draw(self, screen):
        rotated_image = pygame.transform.rotate(self.image, -self.angle * 180 / math.pi)
        rect = rotated_image.get_rect()
        screen.blit(rotated_image, (self.x - rect.width / 2, self.y - rect.height / 2))
