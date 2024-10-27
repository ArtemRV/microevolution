import numpy as np
import pygame

# Настройки
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 800

GRID_SIZE = 50  # Размер ячеек сетки

RADIUS_DISH = 350
OBSTACLE_RADIUS = 10
ORGANISM_RADIUS = 15
FOOD_RADIUS = 5

OBSTACLE_QUANTITY = 50
FOOD_QUANTITY = 600

CENTER = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

class Organism:
    def __init__(self):
        self.x = CENTER[0]
        self.y = CENTER[1]
        self.radius = ORGANISM_RADIUS


class Obstacle:
    def __init__(self, organism, obstacles):
        self.radius = OBSTACLE_RADIUS
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, RADIUS_DISH - self.radius)
            x = CENTER[0] + radius * np.cos(angle)
            y = CENTER[1] + radius * np.sin(angle)

            if not check_collision(x, y, self.radius, organism.x, organism.y, organism.radius):
                if not any(check_collision(x, y, self.radius, o.x, o.y, self.radius) for o in obstacles):
                    self.x, self.y = x, y
                    self.vx, self.vy = np.random.uniform(-2, 2, 2)
                    break

    def move(self):
        # Проверка столкновений с чашкой Петри и другими препятствиями
        if check_dish_collision(self.x + self.vx, self.y + self.vy, self.radius):
            angle_rebound = calculata_rebound_angle(self.x, self.y, CENTER[0], CENTER[1], self.vx, self.vy)
            self.vx = -np.cos(angle_rebound) * np.hypot(self.vx, self.vy)
            self.vy = -np.sin(angle_rebound) * np.hypot(self.vx, self.vy)

        self.x += self.vx
        self.y += self.vy

        nearby = get_nearby_objects(self)
        check_repel(self, nearby)

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)


class Food:
    def __init__(self, obstacles, organism, foods):
        self.radius = FOOD_RADIUS
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, RADIUS_DISH - self.radius)
            x = CENTER[0] + radius * np.cos(angle)
            y = CENTER[1] + radius * np.sin(angle)
            if not check_collision(x, y, self.radius, organism.x, organism.y, organism.radius):
                collision = False
                for obstacle in obstacles:
                    if check_collision(x, y, self.radius, obstacle.x, obstacle.y, obstacle.radius):
                        collision = True
                        break
                if not collision:
                    if foods:
                        for food in foods:
                            if food == self:
                                continue
                            if check_collision(x, y, self.radius, food.x, food.y, self.radius):
                                collision = True
                                break
                    if not collision:
                        self.x, self.y = x, y
                        break               

    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), self.radius)

    def check_eaten(self, organism):
        # Проверка столкновения между организмом и едой
        distance = np.hypot(self.x - organism.x, self.y - organism.y)
        return distance < (self.radius + organism.radius)


def add_to_grid(obj):
    grid_x = int(obj.x // GRID_SIZE)
    grid_y = int(obj.y // GRID_SIZE)
    grid.setdefault((grid_x, grid_y), []).append(obj)

def calculata_rebound_angle(x1, y1, x2, y2, vx, vy):
    angle = np.arctan2(y1 - y2, x1 - x2)
    vel_angle = np.arctan2(vy, vx)
    rebound_angle = 2 * angle - vel_angle
    return rebound_angle

def get_nearby_objects(obj):
    grid_x = int(obj.x // GRID_SIZE)
    grid_y = int(obj.y // GRID_SIZE)
    nearby = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nearby.extend(grid.get((grid_x + dx, grid_y + dy), []))
    return nearby

def check_dish_collision(x, y, radius):
    return calculate_distance(x, y, CENTER[0], CENTER[1]) + radius > RADIUS_DISH

def calculate_distance(x1, y1, x2, y2):
    return np.hypot(x1 - x2, y1 - y2)

# Функция для проверки столкновения
def check_collision(x1, y1, r1, x2, y2, r2):
    distance = np.hypot(x2 - x1, y2 - y1)
    return distance < r1 + r2

# Функция для отталкивания объектов
def repel(moving_obj, static_obj, nearby):
    dx, dy = static_obj.x - moving_obj.x, static_obj.y - moving_obj.y
    distance = np.hypot(dx, dy)
    overlap = moving_obj.radius + static_obj.radius - distance

    # Отталкиваем объект на величину overlap в направлении от движущегося объекта
    if distance != 0:
        direction = np.array([dx / distance, dy / distance])  # Нормализованный вектор
        if check_dish_collision(static_obj.x + direction[0] * overlap, static_obj.y + direction[1] * overlap, static_obj.radius):
            # Угол к центру чашки Петри
            angle_to_center = np.arctan2(static_obj.y - CENTER[1], static_obj.x - CENTER[0])

            # Касательное направление (перпендикулярное к радиальному)
            tangent_dx = -np.sin(angle_to_center)
            tangent_dy = np.cos(angle_to_center)

            # Определяем направление сдвига вдоль касательной
            # Если скалярное произведение положительно — оставляем направление, если отрицательно — инвертируем
            if np.dot(direction, [tangent_dx, tangent_dy]) < 0:
                tangent_dx, tangent_dy = -tangent_dx, -tangent_dy

            # Максимальное допустимое расстояние с учетом радиуса
            max_distance = RADIUS_DISH - static_obj.radius
            static_obj.x = CENTER[0] + max_distance * np.cos(angle_to_center) + tangent_dx * overlap
            static_obj.y = CENTER[1] + max_distance * np.sin(angle_to_center) + tangent_dy * overlap
        else:    
            static_obj.x += direction[0] * overlap
            static_obj.y += direction[1] * overlap
        check_repel(static_obj, nearby)

def check_repel(obj, nearby):
    for other in nearby[:]:
        if other == obj:
            continue
        if check_collision(obj.x, obj.y, obj.radius, other.x, other.y, other.radius):
            if obj in nearby:
                nearby.remove(obj)
            repel(obj, other, nearby)

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Начальные позиции объектов
organism = Organism()
grid = {}

obstacles = []
for _ in range(OBSTACLE_QUANTITY):
    obstacles.append(Obstacle(organism, obstacles))

foods = []
for _ in range(FOOD_QUANTITY):
    foods.append(Food(obstacles, organism, foods))

# Игровой цикл
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    grid.clear()
    for obstacle in obstacles:
        add_to_grid(obstacle)
    for food in foods:
        add_to_grid(food)

    # Очистка экрана и рисование объектов
    screen.fill(WHITE)
    for obstacle in obstacles:
        obstacle.move()
        obstacle.draw(screen)

    for food in foods:
        food.draw(screen)

    pygame.draw.circle(screen, BLUE, CENTER, RADIUS_DISH, 2)

    pygame.display.flip()
    pygame.time.delay(30)

pygame.quit()
