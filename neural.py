import numpy as np
import pygame
import math
from collections import deque

pygame.init()

# Параметры окна
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Petri Dish Simulation")

# Основные цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Радиус чашки Петри и объектов
RADIUS_DISH = 350
OBSTACLE_RADIUS = 10
ORGANISM_RADIUS = 15
FOOD_RADIUS = 5

OBSTACLE_QUANTITY = 15
FOOD_QUANTITY = 30

# Центр чашки Петри
CENTER = (WIDTH // 2, HEIGHT // 2)

INPUT_DIM = 12
OUTPUT_DIM = 2
HIDDEN_DIM = 20
LEARNING_RATE = 0.01

# Font and clock setup
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

# Обобщенные функции
def generate_random_position(radius):
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(0, RADIUS_DISH - radius)
    x = CENTER[0] + dist * np.cos(angle)
    y = CENTER[1] + dist * np.sin(angle)
    return x, y

# Класс для препятствий
class Obstacle:
    def __init__(self):
        self.x, self.y = generate_random_position(OBSTACLE_RADIUS)
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-2, 2)

    def move(self):
        if check_dish_collision(self.x + self.vx, self.y + self.vy, OBSTACLE_RADIUS):
            self.vx, self.vy = reflect_velocity(self.x, self.y, CENTER[0], CENTER[1], self.vx, self.vy)
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), OBSTACLE_RADIUS)

# Класс для пищи
class Food:
    def __init__(self):
        self.x, self.y = generate_random_position(FOOD_RADIUS)

    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), FOOD_RADIUS)

    def regenerate(self):
        self.x, self.y = generate_random_position(FOOD_RADIUS)

    def check_eaten(self, organism):
        distance = np.hypot(self.x - organism.x, self.y - organism.y)
        return distance < (FOOD_RADIUS + ORGANISM_RADIUS)

# Класс для организма
class Organism:
    def __init__(self):
        self.x = CENTER[0]
        self.y = CENTER[1]
        self.speed = 2
        self.vx = np.random.uniform(-self.speed, self.speed)
        self.vy = np.random.uniform(-self.speed, self.speed)
        self.energy = 10
        self.w1, self.b1, self.w2, self.b2 = neural_init()
        self.optimizer_w1 = AdamOptimizer(lr=LEARNING_RATE)
        self.optimizer_b1 = AdamOptimizer(lr=LEARNING_RATE)
        self.optimizer_w2 = AdamOptimizer(lr=LEARNING_RATE)
        self.optimizer_b2 = AdamOptimizer(lr=LEARNING_RATE)
        self.input_vector = None
        self.experience_buffer = deque(maxlen=100)

    def update_weights(self, reward):
        grad_output = np.random.randn(*self.w1.shape) * reward
        self.w1 = self.optimizer_w1.update(self.w1, grad_output)
        self.b1 = self.optimizer_b1.update(self.b1, grad_output)
        self.w2 = self.optimizer_w2.update(self.w2, grad_output)
        self.b2 = self.optimizer_b2.update(self.b2, grad_output)

    def move(self, obstacles, foods):
        closest_food = min(foods, key=lambda food: np.hypot(self.x - food.x, self.y - food.y))
        closest_obstacles = sorted(obstacles, key=lambda obstacle: np.hypot(self.x - obstacle.x, self.y - obstacle.y))[:3]

        self.input_vector = np.array([
            self.x - closest_food.x, self.y - closest_food.y,
            self.x - closest_obstacles[0].x, self.y - closest_obstacles[0].y,
            self.x - closest_obstacles[1].x, self.y - closest_obstacles[1].y,
            self.x - closest_obstacles[2].x, self.y - closest_obstacles[2].y,
            self.vx, self.vy, self.energy,
            np.hypot(self.x - CENTER[0], self.y - CENTER[1]) + ORGANISM_RADIUS - RADIUS_DISH
        ])

        self.vx, self.vy = forward_pass(self.input_vector, self.w1, self.b1, self.w2, self.b2)[:2]
        self.vx *= self.speed
        self.vy *= self.speed
        self.x += self.vx
        self.y += self.vy

        if check_dish_collision(self.x, self.y, ORGANISM_RADIUS):
            self.vx, self.vy = reflect_velocity(self.x, self.y, CENTER[0], CENTER[1], self.vx, self.vy)
            self.correct_position_within_dish()
            self.update_weights(-25)

        for obstacle in obstacles:
            if check_collision(self.x, self.y, ORGANISM_RADIUS, obstacle.x, obstacle.y, OBSTACLE_RADIUS):
                self.vx, self.vy = reflect_velocity(self.x, self.y, obstacle.x, obstacle.y, self.vx, self.vy)
                self.update_weights(-50)

        self.energy -= 0.01 + np.hypot(self.vx, self.vy) * 0.001
        if self.energy <= 0:
            self.reset(-100)
        else:
            self.update_weights(-0.1)

    def correct_position_within_dish(self):
        distance_to_center = np.hypot(self.x - CENTER[0], self.y - CENTER[1])
        overlap = distance_to_center + ORGANISM_RADIUS - RADIUS_DISH
        if overlap > 0:
            angle_rebound = np.arctan2(self.y - CENTER[1], self.x - CENTER[0])
            self.x -= overlap * np.cos(angle_rebound)
            self.y -= overlap * np.sin(angle_rebound)

    def reset(self, reward):
        self.x, self.y = generate_random_position(ORGANISM_RADIUS)
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-2, 2)
        self.energy = 10
        self.update_weights(reward)

# Оптимизация нейросетевого обучения: градиентный спуск Adam
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, dw):
        if self.m is None:
            self.m = np.zeros_like(dw)
            self.v = np.zeros_like(dw)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dw
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dw ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        w -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w

# Функции для рисования и физики
def draw_dish():
    pygame.draw.circle(screen, BLUE, CENTER, RADIUS_DISH, 2)

def check_dish_collision(x, y, radius):
    return np.hypot(x - CENTER[0], y - CENTER[1]) + radius > RADIUS_DISH

def check_collision(x1, y1, r1, x2, y2, r2):
    return np.hypot(x1 - x2, y1 - y2) < r1 + r2

def reflect_velocity(x1, y1, x2, y2, vx, vy):
    angle = np.arctan2(y1 - y2, x1 - x2)
    vel_angle = np.arctan2(vy, vx)
    rebound_angle = 2 * angle - vel_angle
    new_vx = -np.cos(rebound_angle) * np.hypot(vx, vy)
    new_vy = -np.sin(rebound_angle) * np.hypot(vx, vy)
    return new_vx, new_vy

# Текстовый вывод
def render_text(screen, text, x, y):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        rendered_text = font.render(line, True, BLACK)
        screen.blit(rendered_text, (x, y + i * 24))

# Инициализация весов нейросети
def neural_init():
    w1 = np.random.randn(INPUT_DIM, HIDDEN_DIM)
    b1 = np.random.randn(HIDDEN_DIM)
    w2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM)
    b2 = np.random.randn(OUTPUT_DIM)
    return w1, b1, w2, b2

# Прямой проход через нейросеть
def forward_pass(x, w1, b1, w2, b2):
    z1 = np.dot(x, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    output = np.tanh(z2)
    return output

# Основной игровой цикл
def main():
    obstacles = [Obstacle() for _ in range(OBSTACLE_QUANTITY)]
    foods = [Food() for _ in range(FOOD_QUANTITY)]
    organism = Organism()

    running = True
    while running:
        screen.fill(WHITE)
        draw_dish()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        organism.move(obstacles, foods)

        for food in foods:
            if food.check_eaten(organism):
                organism.energy += 3
                food.regenerate()
            food.draw(screen)

        for obstacle in obstacles:
            obstacle.move()
            obstacle.draw(screen)

        pygame.draw.circle(screen, BLACK, (int(organism.x), int(organism.y)), ORGANISM_RADIUS)

        render_text(screen, f"Energy: {organism.energy:.2f}", 10, 10)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
