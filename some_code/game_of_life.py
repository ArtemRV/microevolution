import pygame
import random
import copy
from Utils.text import SimpleText

# Define the grid size and cell size
WIDTH, HEIGHT = 1000, 600
CELL_SIZE = 10
CELL_QUANTITY = 5
BORDER_INDENT = 100
INFO_INDENT = 350
GRID_WIDTH = (WIDTH - BORDER_INDENT * 2 - INFO_INDENT) // CELL_SIZE
GRID_HEIGHT = (HEIGHT - BORDER_INDENT * 2) // CELL_SIZE
MAX_ENERGY = 10
GENOME_SIZE = 20

directions = [
    (1, 0),   # action 0
    (1, 1),   # action 1
    (0, 1),   # action 2
    (-1, 1),  # action 3
    (-1, 0),  # action 4
    (-1, -1), # action 5
    (0, -1),  # action 6
    (1, -1)   # action 7
]

colors = Colors()

class Grid:
    def __init__(self, width, height, border_indent, cell_size) -> None:
        self.width = width
        self.height = height
        self.border_indent = border_indent
        self.indent = 1
        self.grid = self.initialize_grid()

    def initialize_grid(self):
        return [[random.randrange(0, MAX_ENERGY, 1) for _ in range(self.width)] for _ in range(self.height)]
    
    def draw_grid(self, screen):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                color = (25.5 * self.grid[y][x], 25.5 * self.grid[y][x], 25.5 * self.grid[y][x])
                pygame.draw.rect(screen, color, (x * CELL_SIZE + self.border_indent + self.indent, y * CELL_SIZE + self.border_indent + self.indent, CELL_SIZE - 2 * self.indent, CELL_SIZE - 2 * self.indent))


class Cell:
    def __init__(self, name, x, y, energy, color=colors.RED) -> None:
        self.name = name
        self.generations = 0
        self.x = x
        self.y = y
        self.energy = energy
        self.max_energy = 10
        self.max_eating = 5
        self.active_genome = 0
        self.genome = [random.randrange(0, 64, 1) for _ in range(GENOME_SIZE)]
        self.color = color

    def draw_cell(self, screen):
        pygame.draw.rect(screen, self.color, (self.x * CELL_SIZE + BORDER_INDENT, self.y * CELL_SIZE + BORDER_INDENT, CELL_SIZE, CELL_SIZE))


class Population:
    def __init__(self) -> None:
        self.population = []
        for _ in range(CELL_QUANTITY):
            name = 'cell' + str(_)
            x, y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            cell = Cell(name, x, y, 5)
            self.add_cell(cell)

    def add_cell(self, cell):
        self.population.append(cell)

    def remove_cell(self, cell):
        self.population.remove(cell)

    def draw_population(self, screen):
        for cell in self.population:
            cell.draw_cell(screen)    


class Logic:
    def __init__(self) -> None:
        self.movement = list(range(0, 8))
        self.observation = list(range(8, 16))
        self.eating = list(range(16, 24))
        self.reproduction = list(range(24, 32))

    def check_status(self, cell, grid, population):
        print(cell.energy)
        if cell.energy <= 0:
            self.die(cell, grid, population)
        else:
            cell.energy -= 0.3
            action = cell.genome[cell.active_genome]
            if action in self.movement:
                self.move(cell, action)
            elif action in self.observation:
                self.observe(cell, action)
            elif action in self.eating:
                self.eat(cell, action, grid)
            elif action in self.reproduction:
                self.reproduce(cell, action, population)
            cell.active_genome += 1 if cell.active_genome < GENOME_SIZE - 1 else -(GENOME_SIZE - 1)

    def die(self, cell, grid, population):
        population.remove_cell(cell)
        if grid[cell.y][cell.x] + 2 <= MAX_ENERGY:
            grid[cell.y][cell.x] += 2
        else:
            grid[cell.y][cell.x] = MAX_ENERGY

    def move(self, cell, action):       
        dx, dy = directions[action]
        cell.x += dx
        cell.y += dy

        cell.x, cell.y = check_boundaries(cell.x, cell.y)

    def observe(self, cell, action):
        pass

    def eat(self, cell, action, grid):
        if grid[cell.y][cell.x] > 0:
            if grid[cell.y][cell.x] > cell.max_eating:
                portion = cell.max_eating - cell.energy
                cell.energy = cell.max_eating
                grid[cell.y][cell.x] -= portion
            else:
                cell.energy += grid[cell.y][cell.x]
                grid[cell.y][cell.x] = 0

    def reproduce(self, cell, action, population):
        # Energy check
        if cell.energy < 2:
            return
        cell.energy /= 2
        # Reproduction space check
        dx, dy = directions[action - 24]
        x, y = check_boundaries(cell.x + dx, cell.y + dy)
        new_cell = copy.deepcopy(cell)
        new_cell.x, new_cell.y = x, y
        a = random.choice([40, -40])
        b = random.choice([0, 1, 2])
        new_color = list(cell.color)  # Convert to list if it's a tuple
        if new_color[b] + a > 255 or new_color[b] + a < 0:
            new_color[b] -= a
        else:
            new_color[b] += a
        new_cell.color = tuple(new_color)
        new_cell.active_genome = 0
        # if mutation_chance := random.choice([True, False]):
            # new_cell.generations = 0
            # new_cell.name = cell.name + '_g' + str(cell.generations)
        new_cell.generations += 1
        population.add_cell(new_cell)

    def check_free_space(x, y, population):
        for cell in population:
            if cell.x == x and cell.y == y:
                return False
        return True


class Statistics:
    def __init__(self, population) -> None:
        self.statistics = {}
        self.initial_population = {
            cell.name: {
                'genome': cell.genome,
                'generations': cell.generations
            } for cell in population
        }

    def get_statistics(self, population):
        for cell in population:
            if cell.name not in self.statistics or cell.generations > self.statistics[cell.name]['generations']:
                self.statistics[cell.name] = {
                    'genome': cell.genome,
                    'generations': cell.generations
                }

    def draw_statistics(self, population, screen):
        height = 10
        for cell in population:
            text = SimpleText(f"{cell.name}     energy: {round(cell.energy, 2)}", colors)
            text.draw_text((screen.get_width() - INFO_INDENT, height), screen)
            height += 10

    def draw_final_statistics(self, screen):
        screen.fill(colors.BLACK)
        height = 10
        for key, value in self.initial_population.items():
            text = SimpleText(f"{key}  gen: {value['generations']}      genome: {value['genome']}", colors)
            text.draw_text((10, height), screen)
            height += 10
        
        height = 100
        for key, value in self.statistics.items():
            text = SimpleText(f"{key}  gen: {value['generations']}      genome: {value['genome']}", colors)
            text.draw_text((10, height), screen)
            height += 10        
        
        pygame.display.flip()


def check_boundaries(x, y):
    if x > GRID_WIDTH - 1:
        x = 0
    elif x < 0:
        x = GRID_WIDTH - 1
    
    if y > GRID_HEIGHT - 1:
        y = 0
    elif y < 0:
        y = GRID_HEIGHT - 1

    return x, y


# Run the Game of Life with Pygame
def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Game of evloution")
    clock = pygame.time.Clock()

    grid = Grid(GRID_WIDTH, GRID_HEIGHT, BORDER_INDENT, CELL_SIZE)
    logic = Logic()
    population = Population()
    statistics = Statistics(population.population)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(colors.BLACK)

        if population.population:
            grid.draw_grid(screen)
            for cell in population.population:
                logic.check_status(cell, grid.grid, population)

            population.draw_population(screen)

            statistics.get_statistics(population.population)
            statistics.draw_statistics(population.population, screen)

        else:
            statistics.draw_final_statistics(screen)

        pygame.display.flip()
        clock.tick(10)  # Adjust the speed of the game

    pygame.quit()

if __name__ == "__main__":
    run_game()