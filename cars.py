import pygame
import os

from Class.Car import Car
from Class.Map import Map
from Class.MapDesignTools import MapDesignTools

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def draw_level(screen, clock):
    dt = 0
    level_map = Map((WINDOW_WIDTH, WINDOW_HEIGHT))
    map_design_tools = MapDesignTools((WINDOW_WIDTH, WINDOW_HEIGHT))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            map_design_tools.selection = 0
        if keys[pygame.K_2]:
            map_design_tools.selection = 1
        if keys[pygame.K_3]:
            map_design_tools.selection = 2
        if keys[pygame.K_4]:
            map_design_tools.selection = 3
        if keys[pygame.K_5]:
            map_design_tools.selection = 4
        if keys[pygame.K_SPACE]:
            if level_map.ready():
                level_map.compute_mask()
                return level_map

        screen.fill((200, 200, 0))
        level_map.draw(screen)
        current_click = pygame.mouse.get_pressed()[0]
        if current_click:
            if map_design_tools.selection == 3:
                level_map.setFlag(pygame.mouse.get_pos(), "start")
            elif map_design_tools.selection == 4:
                level_map.setFlag(pygame.mouse.get_pos(), "end")
            else:
                pos = pygame.mouse.get_pos()
                level_map.add_circle(pos[0], pos[1], map_design_tools.getRadius())
        else:
            map_design_tools.draw(screen, level_map.start, level_map.end)

        pygame.display.flip()
        dt = clock.tick(60) / 1000


def run_game(screen, clock, level):
    dt = 0

    car = Car(level.start.x, level.start.y, 0)

    pygame.mouse.set_visible(False)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((200, 200, 0))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            car.turn_left()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            car.turn_right()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            car.accelerate()
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            car.decelerate()

        car.update(dt, level)

        level.draw(screen)
        car.draw(screen)

        car.update_distance(level)
        car.draw_distance(screen)

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
    level = draw_level(screen, clock)
    run_game(screen, clock, level)
