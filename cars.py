import pygame
import os
import neat

from Class.Car import Car
from Class.Map import Map
from Class.MapDesignTools import MapDesignTools

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

PATH_NEAT_CONFIG = "neat_config"

best_time = 0


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

        if car.finished:
            break

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

        distances = car.get_distances(level, screen)

        # print the speed of car on top left
        font = pygame.font.SysFont("Arial", 20)
        text = font.render("Speed: {:.2f}".format(car.speed), True, (0, 0, 0))
        screen.blit(text, (10, 10))

        for distance in distances:
            text = font.render(
                "Distance {}: {:.2f}".format(distances.index(distance) + 1, distance),
                True,
                (0, 0, 0),
            )
            screen.blit(text, (10, 10 + 20 * (distances.index(distance) + 1)))

        pygame.display.flip()
        dt = clock.tick(60) / 1000


def eval_genomes_with_level(level):
    screen, clock = init_pygame()

    def eval_genomes(genomes, config):
        dt = 0
        ge = []
        nets = []
        cars = []

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            cars.append(Car(level.start.x, level.start.y, 0))
            g.fitness = 0
            ge.append(g)

        timer = 0
        global best_time

        time_per_generation = 10
        font1 = pygame.font.SysFont("Arial", 50)
        font2 = pygame.font.SysFont("Arial", 30)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            screen.fill((200, 0, 200))

            # TODO CHECK IF ALL CARS ARE FINISHED
            if all(car.finished for car in cars):
                break

            for car in cars:
                car.update(dt, level)

            level.draw(screen)

            for car in cars:
                car.draw(screen)

            time_text = font1.render(
                "{:.2f}".format(time_per_generation - timer), True, (240, 240, 240)
            )
            screen.blit(time_text, (WINDOW_WIDTH / 2 - 10, 10))

            if best_time:
                best_time_text = font2.render(
                    "Best time: {:.4f}".format(best_time), True, (240, 240, 240)
                )
                # on top left
                screen.blit(best_time_text, (10, 10))

            for i, car in enumerate(cars):
                # if car is finished, give it a bonus
                if car.finished:
                    ge[i].fitness += 1000
                    if timer < best_time or best_time == 0:
                        best_time = timer
                        ge[i].fitness += 500
                    return

                # more close of end
                ge[i].fitness += 1 / (
                    abs(car.x - level.end.x) + abs(car.y - level.end.y)
                )
                # if car collide give malus
                if car.speed == 0:
                    ge[i].fitness -= 10

                if car.collided:
                    ge[i].fitness -= 100
                    # remove car
                    cars.pop(i)
                    if len(cars) == 0:
                        return

                distances = car.get_distances(level, screen)
                outputs = nets[i].activate(
                    (
                        distances[0],
                        distances[1],
                        distances[2],
                        distances[3],
                        distances[4],
                        car.speed,
                    )
                )

                if outputs[0] > 0.5:
                    car.turn_left()
                if outputs[1] > 0.5:
                    car.turn_right()
                if outputs[2] > 0.5:
                    car.accelerate()
                if outputs[3] > 0.5:
                    car.decelerate()

            # incr timer
            timer += dt
            if timer > time_per_generation:
                return

            pygame.display.flip()
            dt = clock.tick(60) / 1000

    return eval_genomes


def init_neat(path, level):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path,
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    eval_func = eval_genomes_with_level(level)
    p.run(eval_func, 100)

    pygame.quit()


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Cars")
    clock = pygame.time.Clock()
    return screen, clock


if __name__ == "__main__":
    if True:
        # close screen
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, PATH_NEAT_CONFIG)

        screen, clock = init_pygame()
        level = draw_level(screen, clock)
        pygame.quit()

        init_neat(config_path, level)
    else:
        screen, clock = init_pygame()
        level = draw_level(screen, clock)
        run_game(screen, clock, level)
