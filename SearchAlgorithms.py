#!/usr/bin/env python3
from typing import List, Set
from dataclasses import dataclass
import pygame
from enum import Enum, unique
import sys
import random


FPS = 30

INIT_LENGTH = 4

WIDTH = 480
HEIGHT = 480
GRID_SIDE = 24
GRID_WIDTH = WIDTH // GRID_SIDE
GRID_HEIGHT = HEIGHT // GRID_SIDE

BRIGHT_BG = (103, 223, 235)
DARK_BG = (78, 165, 173)

SNAKE_COL = (6, 38, 7)
FOOD_COL = (224, 160, 38)
OBSTACLE_COL = (209, 59, 59)
VISITED_COL = (24, 42, 142)


@unique
class Direction(tuple, Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    def reverse(self):
        x, y = self.value
        return Direction((x * -1, y * -1))


@dataclass
class Position:
    x: int
    y: int

    def check_bounds(self, width: int, height: int):
        return (self.x >= width) or (self.x < 0) or (self.y >= height) or (self.y < 0)

    def draw_node(self, surface: pygame.Surface, color: tuple, background: tuple):
        r = pygame.Rect(
            (int(self.x * GRID_SIDE), int(self.y * GRID_SIDE)), (GRID_SIDE, GRID_SIDE)
        )
        pygame.draw.rect(surface, color, r)
        pygame.draw.rect(surface, background, r, 1)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Position):
            return (self.x == o.x) and (self.y == o.y)
        else:
            return False

    def __str__(self):
        return f"X{self.x};Y{self.y};"

    def __hash__(self):
        return hash(str(self))


class GameNode:
    nodes: Set[Position] = set()

    def __init__(self):
        self.position = Position(0, 0)
        self.color = (0, 0, 0)

    def randomize_position(self):
        try:
            GameNode.nodes.remove(self.position)
        except KeyError:
            pass

        condidate_position = Position(
            random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1),
        )

        if condidate_position not in GameNode.nodes:
            self.position = condidate_position
            GameNode.nodes.add(self.position)
        else:
            self.randomize_position()

    def draw(self, surface: pygame.Surface):
        self.position.draw_node(surface, self.color, BRIGHT_BG)


class Food(GameNode):
    def __init__(self):
        super(Food, self).__init__()
        self.color = FOOD_COL
        self.randomize_position()


class Obstacle(GameNode):
    def __init__(self):
        super(Obstacle, self).__init__()
        self.color = OBSTACLE_COL
        self.randomize_position()


class Snake:
    def __init__(self, screen_width, screen_height, init_length):
        self.color = SNAKE_COL
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.init_length = init_length
        self.reset()

    def reset(self):
        self.length = self.init_length
        self.positions = [Position((GRID_SIDE // 2), (GRID_SIDE // 2))]
        self.direction = random.choice([e for e in Direction])
        self.score = 0
        self.hasReset = True

    def get_head_position(self) -> Position:
        return self.positions[0]

    def turn(self, direction: Direction):
        if self.length > 1 and direction.reverse() == self.direction:
            return
        else:
            self.direction = direction

    def move(self):
        self.hasReset = False
        cur = self.get_head_position()
        x, y = self.direction.value
        new = Position(cur.x + x, cur.y + y,)
        if self.collide(new):
            self.reset()
        else:
            self.positions.insert(0, new)
            while len(self.positions) > self.length:
                self.positions.pop()

    def collide(self, new: Position):
        return (new in self.positions) or (new.check_bounds(GRID_WIDTH, GRID_HEIGHT))

    def eat(self, food: Food):
        if self.get_head_position() == food.position:
            self.length += 1
            self.score += 1
            while food.position in self.positions:
                food.randomize_position()

    def hit_obstacle(self, obstacle: Obstacle):
        if self.get_head_position() == obstacle.position:
            self.length -= 1
            self.score -= 1
            if self.length == 0:
                self.reset()

    def draw(self, surface: pygame.Surface):
        for p in self.positions:
            p.draw_node(surface, self.color, BRIGHT_BG)


class Player:
    def __init__(self) -> None:
        self.visited_color = VISITED_COL
        self.visited: Set[Position] = set()
        self.chosen_path: List[Direction] = []

    def move(self, snake: Snake) -> bool:
        try:
            next_step = self.chosen_path.pop(0)
            snake.turn(next_step)
            return False
        except IndexError:
            return True

    def search_path(self, snake: Snake, food: Food, *obstacles: Set[Obstacle]):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def turn(self, direction: Direction):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def draw_visited(self, surface: pygame.Surface):
        for p in self.visited:
            p.draw_node(surface, self.visited_color, BRIGHT_BG)


class SnakeGame:
    def __init__(self, snake: Snake, player: Player) -> None:
        pygame.init()
        pygame.display.set_caption("AIFundamentals - SnakeGame")

        self.snake = snake
        self.food = Food()
        self.obstacles: Set[Obstacle] = set()
        for _ in range(40):
            ob = Obstacle()
            while any([ob.position == o.position for o in self.obstacles]):
                ob.randomize_position()
            self.obstacles.add(ob)

        self.player = player

        self.fps_clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(
            (snake.screen_height, snake.screen_width), 0, 32
        )
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        self.myfont = pygame.font.SysFont("monospace", 16)

    def drawGrid(self):
        for y in range(0, int(GRID_HEIGHT)):
            for x in range(0, int(GRID_WIDTH)):
                p = Position(x, y)
                if (x + y) % 2 == 0:
                    p.draw_node(self.surface, BRIGHT_BG, BRIGHT_BG)
                else:
                    p.draw_node(self.surface, DARK_BG, DARK_BG)

    def run(self):
        while not self.handle_events():
            self.fps_clock.tick(FPS)
            self.drawGrid()
            if self.player.move(self.snake) or self.snake.hasReset:
                self.player.search_path(self.snake, self.food, self.obstacles)
                self.player.move(self.snake)
            self.snake.move()
            self.snake.eat(self.food)
            for ob in self.obstacles:
                self.snake.hit_obstacle(ob)
            for ob in self.obstacles:
                ob.draw(self.surface)
            self.player.draw_visited(self.surface)
            self.snake.draw(self.surface)
            self.food.draw(self.surface)
            self.screen.blit(self.surface, (0, 0))
            text = self.myfont.render(
                "Score {0}".format(self.snake.score), 1, (0, 0, 0)
            )
            self.screen.blit(text, (5, 10))
            pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_UP:
                    self.player.turn(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self.player.turn(Direction.DOWN)
                elif event.key == pygame.K_LEFT:
                    self.player.turn(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.player.turn(Direction.RIGHT)
        return False


class HumanPlayer(Player):
    def __init__(self):
        super(HumanPlayer, self).__init__()

    def turn(self, direction: Direction):
        self.chosen_path.append(direction)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------


from collections import deque

class SearchBasedPlayer(Player):
    def __init__(self, algorithm="A*"):
        super(SearchBasedPlayer, self).__init__()
        self.algorithm = algorithm

    def search_path(self, snake: Snake, food: Food, obstacles: Set[Obstacle]):
        if self.algorithm == "BFS":
            self.bfs_search(snake, food, obstacles)
        elif self.algorithm == "DFS":
            self.dfs_search(snake, food, obstacles)
        elif self.algorithm == "Dijkstra":
            self.dijkstra_search(snake, food, obstacles)
        elif self.algorithm == "A*":
            self.a_star_search(snake, food, obstacles)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def bfs_search(self, snake: Snake, food: Food, obstacles: Set[Obstacle]):
        self.visited.clear()
        self.chosen_path.clear()

        queue = deque([(snake.get_head_position(), [])])    #initialize queue: stores position to explore and path to get there
        self.visited.add(snake.get_head_position())
        obstacle_positions = {ob.position for ob in obstacles}

        while queue:
            current_position, path = queue.popleft()    #removes first element in queue

            if current_position == food.position:
                self.chosen_path = path
                return

            for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                next_position = Position(
                    current_position.x + direction.value[0],
                    current_position.y + direction.value[1]
                )

                if (
                    next_position in self.visited or
                    next_position in obstacle_positions or
                    next_position in snake.positions or
                    next_position.check_bounds(GRID_WIDTH, GRID_HEIGHT)
                ):
                    continue

                self.visited.add(next_position)
                queue.append((next_position, path + [direction]))   #next position to explore, path to get there + new direction

        self.chosen_path = []   #if didn't fruit, finish with empty path

    def dfs_search(self, snake: Snake, food: Food, obstacles: Set[Obstacle]):
        self.visited.clear()
        self.chosen_path.clear()
        
        stack = [(snake.get_head_position(), [])]   #last in first out - good - because we explore each path fully, if no found backtrack one and try new
        self.visited = [snake.get_head_position()]  #tracks order of visited nodes
        visited_set = {snake.get_head_position()} #fast lookup of visited positions
        
        obstacle_positions = {ob.position for ob in obstacles}
        
        while stack:
            current_position, path = stack.pop(0)
            
            if current_position == food.position:
                self.chosen_path = path
                return
            
            for direction in [Direction.DOWN, Direction.UP, Direction.LEFT, Direction.RIGHT]:
                next_position = Position(
                    current_position.x + direction.value[0],
                    current_position.y + direction.value[1]
                )
                
                if (
                    next_position in visited_set or
                    next_position in obstacle_positions or
                    next_position in snake.positions or
                    next_position.check_bounds(GRID_WIDTH, GRID_HEIGHT)
                ):
                    continue
                
                visited_set.add(next_position)
                self.visited.insert(0, next_position)
                stack.insert(0, (next_position, path + [direction]))

    def dijkstra_search(self, snake: Snake, food: Food, obstacles: Set[Obstacle]):
        self.visited.clear()
        self.chosen_path.clear()

        start_position = snake.get_head_position()
        to_visit = [(0, start_position, [])]    #positions to be checked, current cost, path to get there
        cost_map = {start_position: 0}

        obstacle_positions = {ob.position for ob in obstacles}

        while to_visit:
            to_visit.sort(key=lambda x: x[0])
            current_cost, current_position, path = to_visit.pop(0)

            if current_position == food.position:
                self.chosen_path = path
                return

            self.visited.add(current_position)

            for direction in [Direction.DOWN, Direction.UP, Direction.LEFT, Direction.RIGHT]:
                next_position = Position(
                    current_position.x + direction.value[0],
                    current_position.y + direction.value[1]
                )

                if (
                    next_position in obstacle_positions or
                    next_position in snake.positions or
                    next_position in self.visited or
                    next_position.check_bounds(GRID_WIDTH, GRID_HEIGHT)
                ):
                    continue

                new_cost = current_cost + 1

                if next_position not in cost_map or new_cost < cost_map[next_position]: #check if next_position haven't been reached by cheaper path or check if new cost is lower than previous that led to it
                    cost_map[next_position] = new_cost
                    to_visit.append((new_cost, next_position, path + [direction]))

        self.chosen_path = []
         
    def a_star_search(self, snake: Snake, food: Food, obstacles: Set[Obstacle]):
        self.visited.clear()
        self.chosen_path.clear()

        start_position = snake.get_head_position()
        queue = [(0, start_position, [])] 
        self.visited.add(start_position)
        obstacle_positions = {ob.position for ob in obstacles}

        def heuristic(pos):
            return abs(pos.x - food.position.x) + abs(pos.y - food.position.y)

        while queue:
            queue.sort(key=lambda x: x[0])
            f_score, current_position, path = queue.pop(0)

            if current_position == food.position:
                self.chosen_path = path
                return

            self.visited.add(current_position)

            for direction in [Direction.DOWN, Direction.UP, Direction.LEFT, Direction.RIGHT]:
                next_position = Position(
                    current_position.x + direction.value[0],
                    current_position.y + direction.value[1]
                )

                if (
                    next_position in self.visited or
                    next_position in obstacle_positions or
                    next_position in snake.positions or
                    next_position.check_bounds(GRID_WIDTH, GRID_HEIGHT)
                ):
                    continue

                g_cost = len(path) + 1  #path cost
                h_cost = heuristic(next_position)   #distance cost
                f_score = g_cost + h_cost   #cumulative cost

                queue.append((f_score, next_position, path + [direction]))
                self.visited.add(next_position)

        self.chosen_path = []


if __name__ == "__main__":
    snake = Snake(WIDTH, WIDTH, INIT_LENGTH)
    #player = HumanPlayer()
    player = SearchBasedPlayer()
    game = SnakeGame(snake, player)
    game.run()