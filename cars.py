import pygame
import os

from Class.Car import Car
from Class.Map import Map
from Class.Overlay import Overlay

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def draw_level(screen, clock):
    dt = 0
    level_map = Map()
    overlay = Overlay((WINDOW_WIDTH, WINDOW_HEIGHT))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            level_map.add_circle(pos[0], pos[1], 50)

        screen.fill((0, 0, 0))

        level_map.draw(screen)
        overlay.draw_overlay(screen)

        pygame.display.flip()
        dt = clock.tick(60) / 1000


def run_game(screen, clock):
    dt = 0

    car = Car(100, 100)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((250, 250, 250))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            car.turn_left()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            car.turn_right()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            car.accelerate()
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            car.decelerate()

        car.update(dt)
        car.draw(screen)

        # print the speed of car on top left
        font = pygame.font.SysFont("Arial", 20)
        text = font.render("Speed: {:.2f}".format(car.speed), True, (0, 0, 0))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Cars")
    clock = pygame.time.Clock()
    draw_level(screen, clock)
    run_game(screen, clock)
