import numpy as np
import random
import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, OrderedDict

pygame.init()
torch.autograd.set_detect_anomaly(True)

# Параметры окна
WIDTH, HEIGHT = 1500, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
# screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Micro evolution")

# Основные цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
CUAN = (0, 255, 255)
AZURE = (0, 127, 255)

# Радиус чашки Петри и препятствий
RADIUS_DISH = 350
OBSTACLE_RADIUS = 10
ORGANISM_RADIUS = 15
FOOD_RADIUS = 5

# Размер ячеек сетки
GRID_SIZE = 50

# Количество препятствий и еды
OBSTACLE_QUANTITY = 30
FOOD_QUANTITY = 60

# Центр чашки Петри
CENTER = (screen.get_width() // 2, screen.get_height() // 2)

# Параметры нейронной сети
INPUT_DIM = 16
HIDDEN_DIMS = [32, 16]
OUTPUT_DIM = 2

LEARNING_RATE = 0.0001

# Параметры обучения
EPSILON = 0.1  # Для epsilon-greedy стратегии
GAMMA = 0.99  # Фактор дисконтирования
TAU = 0.005  # Скорость копирования весов основной сети в target сеть

# Font for rendering text
font = pygame.font.SysFont(None, 18)
clock = pygame.time.Clock()

class Obstacle:
    def __init__(self, organism, obstacles):
        self.radius = OBSTACLE_RADIUS
        self.speed = 1
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, RADIUS_DISH - self.radius)
            x = CENTER[0] + radius * np.cos(angle)
            y = CENTER[1] + radius * np.sin(angle)

            if not check_collision(x, y, self.radius, organism.x, organism.y, organism.radius):
                if not any(check_collision(x, y, self.radius, o.x, o.y, self.radius) for o in obstacles):
                    self.x, self.y = x, y
                    self.vx, self.vy = np.random.uniform(-self.speed, self.speed, 2)
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
        

class MatrixPlotter:
    def __init__(self, matrix):
        self.matrix = matrix
        width, height = self.matrix.shape
        self.width = width
        self.height = height
        self.colors = self.generate_colors()
        self.loss = []
        self.max_loss_count = 200
        self.average_loss = 0
        self.average_loss_max = 0

    def generate_colors(self):
        return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(self.width)]

    def update(self, matrix):
        self.matrix = matrix

    def draw(self, screen):
        shift_x = 10
        shift_y = 10
        ratio_x = 20
        ratio_y = 4
        for i in range(self.width):
            for j in range(self.height - 1):
                height1 = self.matrix[i][j] * 10 + (ratio_y * (i + 1)) + shift_y
                height2 = self.matrix[i][j + 1] * 10 + (ratio_y * (i + 1)) + shift_y
                width1 = ratio_x * j + shift_x
                width2 = ratio_x * (j + 1) + shift_x
                pygame.draw.line(screen, self.colors[i], (width1, height1), (width2, height2), 2)

    def update_loss(self, loss):
        if len(self.loss) > self.max_loss_count:
            self.loss.pop(0)
        self.loss.append(loss)
        self.average_loss = np.log(sum(self.loss) / len(self.loss)) * 20
        if self.average_loss > self.average_loss_max:
            self.average_loss_max = self.average_loss

    def draw_loss(self, screen):
        shift_x = 10
        plotter_shift_x = 30
        shift_y = screen.get_height() - 300
        for i, loss_item in enumerate(self.loss):
            if loss_item > 1:
                loss_value = np.log(loss_item) * 20
            else:
                loss_value = loss_item
            pygame.draw.line(screen, RED, (i + shift_x + plotter_shift_x, shift_y), (i + shift_x + plotter_shift_x, shift_y - loss_value), 1)
        # Добавить шкалу логарифмического масштаба
        render_text(screen, "0", shift_x, shift_y)
        pygame.draw.line(screen, BLACK, (shift_x + plotter_shift_x, shift_y), (shift_x + plotter_shift_x + self.max_loss_count, shift_y), 1)
        render_text(screen, "10", shift_x, shift_y - 46)
        pygame.draw.line(screen, BLACK, (shift_x + plotter_shift_x, shift_y - 46), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 46), 1)
        render_text(screen, "100", shift_x, shift_y - 92)
        pygame.draw.line(screen, BLACK, (shift_x + plotter_shift_x, shift_y - 92), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 92), 1)
        render_text(screen, "1000", shift_x, shift_y - 138)
        pygame.draw.line(screen, BLACK, (shift_x + plotter_shift_x, shift_y - 138), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 138), 1)
        # Отрисовка среднего значения
        render_text(screen, f"Av loss: {round(self.average_loss, 2)}", shift_x + plotter_shift_x + self.max_loss_count + 5, shift_y - self.average_loss)
        render_text(screen, f"Max av loss: {round(self.average_loss_max, 2)}", shift_x + plotter_shift_x + self.max_loss_count + 5, shift_y - self.average_loss_max - 10)
        pygame.draw.line(screen, AZURE, (shift_x + plotter_shift_x, shift_y - self.average_loss), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - self.average_loss), 1)
        pygame.draw.line(screen, BLUE, (shift_x + plotter_shift_x, shift_y - self.average_loss_max), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - self.average_loss_max), 1)  


class VectorsPlotter:
    def __init__(self):
        self.vectors = []
        self.start = []
        self.rewards = []

    def update(self, organism, reward):
        self.vectors.append([organism.vx, organism.vy])
        if len(self.start) == 0:
            self.start = [organism.x, organism.y]
        self.rewards.append(reward)

    def draw_vectors_plotter(self, screen):
        a = 20
        x = 100
        draw_center = [screen.get_width() - 400, screen.get_height() // 2 - 100]
        for i, vector in enumerate(self.vectors):
            shift_x = i // a * x
            shift_y = i * a - (i // a) * a**2
            start_pos = [draw_center[0] + shift_x, draw_center[1] + shift_y]
            end_pos = [draw_center[0] + int(vector[0] * 10)  + shift_x, draw_center[1] + int(vector[1] * 10) + shift_y]
            pygame.draw.line(screen, RED, start_pos, end_pos, 2)
            pygame.draw.circle(screen, BLUE, end_pos, 5)
            render_text(screen, f"Rew: {round(self.rewards[i], 2)}", draw_center[0] + 20  + shift_x, draw_center[1] + shift_y)

    def reset(self):
        self.vectors = []
        self.start = []
        self.rewards = []

    def draw_vectors(self, organism, surface):
        shift = organism.radius + 3
        reward_vectors = []

        # Предварительные вычисления для всех направлений
        for alfa in range(0, 360, 6):
            radian_alfa = np.radians(alfa)
            vx = np.cos(radian_alfa)
            vy = np.sin(radian_alfa)
            
            # Подсчет награды от пищи и препятствий
            reward = sum((3 - i) / 3 * calculate_reward(organism.x, organism.y, vx, vy, food.x, food.y, organism.radius, food.radius, 75)
                        for i, food in enumerate(organism.closest_food))
            reward -= sum((3 - i) / 3 * calculate_reward(organism.x, organism.y, vx, vy, obstacle.x, obstacle.y, organism.radius, obstacle.radius, 300)
                        for i, obstacle in enumerate(organism.closest_obstacles))
            
            # Определение цвета и направления
            color = (0, 255, 0, 127) if reward >= 0 else (255, 0, 0, 127)
            reward_vectors.append((vx * abs(reward), vy * abs(reward), color))

        # Рисование всех векторов
        for idx, (vx, vy, color) in enumerate(reward_vectors):
            radian_alfa = np.radians(idx * 6)
            x = organism.x + shift * np.cos(radian_alfa)
            y = organism.y + shift * np.sin(radian_alfa)
            pygame.draw.line(surface, color, (x, y), (x + vx, y + vy), 1)


# Определение архитектуры нейронной сети для DQN
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DQN, self).__init__()
        
        # Создаем список линейных слоев
        layers = []
        current_dim = input_dim

        # Добавляем скрытые слои согласно `hidden_dims`
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh()) # Добавляем функцию активации
            current_dim = hidden_dim

        # Добавляем выходной слой
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Tanh())
        
        # Объединяем слои в nn.Sequential для удобства
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def get_weights(self):
        weights = [layer.weight.data.numpy() for layer in self.model if isinstance(layer, nn.Linear)]
        return weights
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))


# Класс для организма
class Organism:
    def __init__(self):
        self.radius = ORGANISM_RADIUS
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, RADIUS_DISH - self.radius)
        self.x = CENTER[0] + radius * np.cos(angle)
        self.y = CENTER[1] + radius * np.sin(angle)
        self.speed = 1.5
        self.vx = np.random.uniform(-self.speed, self.speed)
        self.vy = np.random.uniform(-self.speed, self.speed)
        self.energy = 10
        self.input_vector = None
        self.reset_status = False
        self.done = False
        self.reset_counters()

    def reset_counters(self):
        self.food_eaten = 0
        self.death_counter = 0
        self.dish_collision_counter = 0
        self.obstacle_collision_counter = 0
        self.energy_loss_counter = 0

    def get_input_vector(self, obstacles, foods):
        # Находим ближайшую еду и препятствия
        closest_food = sorted(foods, key=lambda food: np.hypot(self.x - food.x, self.y - food.y))[:3]
        closest_obstacles = sorted(obstacles, key=lambda obstacle: np.hypot(self.x - obstacle.x, self.y - obstacle.y))[:3]
        
        self.closest_food = closest_food
        self.closest_obstacles = closest_obstacles
        
        # Заполнение входного вектора для нейросети
        self.input_vector = np.array([
            closest_food[0].x - self.x, closest_food[0].y - self.y,
            closest_food[1].x - self.x, closest_food[1].y - self.y,
            closest_food[2].x - self.x, closest_food[2].y - self.y,
            self.vx, self.vy,
            np.hypot(self.x - CENTER[0], self.y - CENTER[1]) + self.radius - RADIUS_DISH,
            self.energy,
            self.x - closest_obstacles[0].x, self.y - closest_obstacles[0].y,
            self.x - closest_obstacles[1].x, self.y - closest_obstacles[1].y,
            self.x - closest_obstacles[2].x, self.y - closest_obstacles[2].y
        ])
        return torch.tensor(self.input_vector, dtype=torch.float32)

    def move(self, action, obstacles, reward, foods):
        self.done = False
        self.reset_status = False
        action_tensor = action.clone().detach().view(-1)  # Ensure action is a PyTorch tensor
        new_vx = action_tensor[0].item() * self.speed  # Directly convert to number
        new_vy = action_tensor[1].item() * self.speed  # Directly convert to number

        # Обновляем позицию организма
        self.x += new_vx
        self.y += new_vy

        # Обновляем скорость
        self.vx, self.vy = new_vx, new_vy

        # Столкновение с границей чашки Петри
        self.dish_collision(reward)
        self.obstacle_collision(reward)
        self.food_collision(reward, obstacles, foods)
        self.energy_update(reward)

        if self.reset_status == True:
            self.reset(obstacles)

        return self.done

    def food_collision(self, reward, obstacles, foods):
        for i, food in enumerate(self.closest_food):
            if food.check_eaten(self):
                self.energy += 5
                foods.remove(food)
                foods.append(Food(obstacles, self, foods))
                new_reward = 30
                self.food_eaten += 1
            else:
                new_reward = (3 - i) / 3 * calculate_reward(self.x, self.y, self.vx, self.vy, food.x, food.y, self.radius, food.radius, 75)
            reward.update(new_reward, 'eat')

    def obstacle_collision(self, reward):
        for i, obstacle in enumerate(self.closest_obstacles):
            if check_collision(self.x, self.y, self.radius, obstacle.x, obstacle.y, obstacle.radius):
                new_reward = -50
                self.obstacle_collision_counter += 1
                self.reset_status = True
            else:
                new_reward = -(3 - i) / 3 * calculate_reward(self.x, self.y, self.vx, self.vy, obstacle.x, obstacle.y, self.radius, obstacle.radius, 300)
                # new_reward = (3 - i) / 3 * calculate_obstacle_collision_reward(self.x, self.y, obstacle.x, obstacle.y, -40)
            reward.update(new_reward, 'obstacle_collision')

    def energy_update(self, reward):
        self.energy -= 0.03 + np.hypot(self.vx, self.vy) * 0.001  # Штраф за скорость
        if self.energy <= 0:
            new_reward = -70
            self.energy_loss_counter += 1
            self.reset_status = True
        else:
            new_reward = calculate_simple_reward(self.energy, -7)
        reward.update(new_reward, 'energy')
    
    def dish_collision(self, reward):
        previous_distance = calculate_distance(self.x, self.y, CENTER[0], CENTER[1]) - RADIUS_DISH + self.radius
        overlap = calculate_distance(self.x + self.vx, self.y + self.vy, CENTER[0], CENTER[1]) - RADIUS_DISH + self.radius
        if overlap > 0:
            new_reward = -30
            self.dish_collision_counter += 1
            self.reset_status = True
        elif overlap > -100 and previous_distance - overlap > 0:
            new_reward = abs(previous_distance - overlap) * 2
        elif overlap > -100 and previous_distance - overlap < 0:
            new_reward = -abs(previous_distance - overlap) * 15
        else:
            new_reward = 0
        reward.update(new_reward, 'dish_collision')


    def reset(self, obstacles):
        """Сброс позиции организма при столкновении с препятствием."""
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, RADIUS_DISH - self.radius)
            self.x = CENTER[0] + radius * np.cos(angle)
            self.y = CENTER[1] + radius * np.sin(angle)
            collision = False
            for obstacle in obstacles:
                if check_collision(self.x, self.y, self.radius, obstacle.x, obstacle.y, obstacle.radius):
                    collision = True
                    break
            if not collision:
                break
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-2, 2)
        self.energy = 10
        self.done = True
        self.death_counter += 1


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


class Reward:
    def __init__(self):
        self.reset()

    def update(self, reward, reward_type):
        if reward_type == 'eat':
            self.eat_reward += reward
        elif reward_type == 'obstacle_collision':
            self.obstacle_collision_reward += reward
        elif reward_type == 'dish_collision':
            self.dish_collision_reward += reward
        elif reward_type == 'energy':
            self.energy_reward += reward
        
        self.reward += reward

    def draw(self, screen):
        shift_x = 10
        shift_y = screen.get_height() - 200
        render_text(screen, f"Total reward:                 {round(self.reward, 2)}", shift_x, shift_y)
        render_text(screen, f"Eat reward:                   {round(self.eat_reward, 2)}", shift_x, shift_y + 10)
        render_text(screen, f"Obstacle collision reward:    {round(self.obstacle_collision_reward, 2)}", shift_x, shift_y + 20)
        render_text(screen, f"Dish collision reward:        {round(self.dish_collision_reward, 2)}", shift_x, shift_y + 30)
        render_text(screen, f"Energy reward:                {round(self.energy_reward, 2)}", shift_x, shift_y + 40)

    def get(self):
        return self.reward

    def reset(self):
        self.reward = 0
        self.eat_reward = 0
        self.obstacle_collision_reward = 0
        self.dish_collision_reward = 0
        self.energy_reward = 0


class Statistics:
    def __init__(self):
        self.episode = OrderedDict()
        self.energy_max = 0
        self.total_reward = 0
        self.total_time = 0
        self.max_episodes = 9

    def update_energy_max(self, energy):
        if energy > self.energy_max:
            self.energy_max = energy

    def update_total_reward(self, reward):
        self.total_reward += reward

    def reset(self):
        self.energy_max = 0
        self.total_reward = 0
        
    def create_episode(self, episode, organism):
        self.reset()
        self.delete_episode()
        self.episode[episode] = {
            'energy max': self.energy_max,
            'food eaten': organism.food_eaten,
            'death counter': organism.death_counter,
            'dish collision counter': organism.dish_collision_counter,
            'obstacle collision counter': organism.obstacle_collision_counter,
            'energy loss counter': organism.energy_loss_counter,
            'reward': self.total_reward,
            'time': self.total_time
        }

    def update_episode(self, episode, organism):
        self.episode[episode]['energy max'] = self.energy_max
        self.episode[episode]['food eaten'] = organism.food_eaten
        self.episode[episode]['death counter'] = organism.death_counter
        self.episode[episode]['dish collision counter'] = organism.dish_collision_counter
        self.episode[episode]['obstacle collision counter'] = organism.obstacle_collision_counter
        self.episode[episode]['energy loss counter'] = organism.energy_loss_counter
        self.episode[episode]['reward'] = self.total_reward
        self.episode[episode]['time'] = self.total_time

    def delete_episode(self):
        if len(self.episode) > self.max_episodes:
            self.episode.popitem(last=False)

    def draw_statistics(self, screen):
        shift_x = 10
        shift_y = screen.get_height() - 100
        episode_list = list(self.episode.items())[-self.max_episodes:]
        data_keys = [
            ("Energy max", "energy max"),
            ("Food eaten", "food eaten"),
            ("Death counter", "death counter"),
            ("Dish collision counter", "dish collision counter"),
            ("Obstacle collision", "obstacle collision counter"),
            ("Energy loss counter", "energy loss counter"),
            ("Total reward", "reward"),
            ("Time", "time")
        ]
        
        for i, (episode, data) in enumerate(episode_list):
            ratio_x = 150
            total_x_shift = shift_x + ratio_x * i
            render_text(screen, f"Episode: {episode}", total_x_shift, shift_y)
            
            for i, (label, key) in enumerate(data_keys):
                value = round(data[key], 2) if isinstance(data[key], float) else data[key]
                render_text(screen, f"{label}: {value}", total_x_shift, shift_y + (i + 1) * 10)

def calculate_simple_reward(parameter, factor):
    calculated_reward = -5 / abs(parameter)
    if calculated_reward < factor:
        calculated_reward = factor
    return calculated_reward

# def calculate_reward(x1, y1, vx, vy, x2, y2, factor):
#     initial_distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
#     new_distance = np.linalg.norm(np.array([x1 + vx, y1 + vy]) - np.array([x2, y2]))
#     distance_diff = initial_distance - new_distance
#     calculated_reward = distance_diff * factor * (1 / new_distance)
#     return calculated_reward

# def calculate_reward(x1, y1, vx, vy, x2, y2, factor):
#     new_distance = np.linalg.norm(np.array([x1 + vx, y1 + vy]) - np.array([x2, y2])) - ORGANISM_RADIUS - FOOD_RADIUS
#     reward = (20 / new_distance ** 2) * 100
#     if reward > 13:
#         reward = 13
#     return reward

def calculate_reward(x1, y1, vx, vy, x2, y2, r1, r2, factor):
    # Вычисляем смещение после шага
    dx, dy = x1 - x2, y1 - y2
    initial_distance = np.hypot(dx, dy)  # Евклидово расстояние

    # Новая позиция после перемещения
    new_x1, new_y1 = x1 + vx, y1 + vy
    new_dx, new_dy = new_x1 - x2, new_y1 - y2
    new_distance = np.hypot(new_dx, new_dy)

    # Награда за изменение расстояния
    distance_diff = initial_distance - new_distance
    vector_reward = distance_diff * factor / new_distance

    # Награда за близость к цели
    adjusted_distance = new_distance - r1 - r2
    distance_reward = (10 / adjusted_distance ** 2) * 100

    # Общая награда с ограничением
    reward = vector_reward# / distance_reward
    # reward = min(vector_reward + distance_reward, 5)
    
    return reward

def calculate_obstacle_collision_reward(target_x, target_y, current_x, current_y, limit):
    distance = np.hypot(target_x - current_x, target_y - current_y) - ORGANISM_RADIUS - OBSTACLE_RADIUS
    reward = (-10 / distance) * 30 + 7
    if reward < limit:
        reward = limit
    if reward > 0:
        reward = 0
    return reward

# Функции для рисования и физики
def add_to_grid(obj):
    grid_x = int(obj.x // GRID_SIZE)
    grid_y = int(obj.y // GRID_SIZE)
    grid.setdefault((grid_x, grid_y), []).append(obj)

def get_nearby_objects(obj):
    grid_x = int(obj.x // GRID_SIZE)
    grid_y = int(obj.y // GRID_SIZE)
    nearby = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nearby.extend(grid.get((grid_x + dx, grid_y + dy), []))
    return nearby

def draw_dish():
    pygame.draw.circle(screen, BLUE, CENTER, RADIUS_DISH, 2)

def calculate_distance(x1, y1, x2, y2):
    return np.hypot(x1 - x2, y1 - y2)

def check_dish_collision(x, y, radius):
    return calculate_distance(x, y, CENTER[0], CENTER[1]) + radius > RADIUS_DISH

def check_collision(x1, y1, r1, x2, y2, r2):
    return np.hypot(x1 - x2, y1 - y2) < r1 + r2

def calculata_rebound_angle(x1, y1, x2, y2, vx, vy):
    angle = np.arctan2(y1 - y2, x1 - x2)
    vel_angle = np.arctan2(vy, vx)
    rebound_angle = 2 * angle - vel_angle
    return rebound_angle

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

# Текстовый вывод
def render_text(screen, text, x, y):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        img = font.render(line, True, BLACK)
        screen.blit(img, (x, y + i * 20))

def get_input_text(input_vector, reward, episode, loss_item):
    text = "Input vector:\n"
    text += f"food_x: {input_vector[0]:.2f} food_y: {input_vector[1]:.2f}\n"
    text += f"food_x: {input_vector[2]:.2f} food_y: {input_vector[3]:.2f}\n"
    text += f"food_x: {input_vector[4]:.2f} food_y: {input_vector[5]:.2f}\n"
    text += f"vx: {input_vector[6]:.2f} vy: {input_vector[7]:.2f}\n"
    text += f"distance_to_boundary: {input_vector[8]:.2f}\n"
    text += f"energy: {input_vector[9]:.2f}\n"
    text += f"obstacle1_x: {input_vector[10]:.2f} obstacle1_y: {input_vector[11]:.2f}\n"
    text += f"obstacle2_x: {input_vector[12]:.2f} obstacle2_y: {input_vector[13]:.2f}\n"
    text += f"obstacle3_x: {input_vector[14]:.2f} obstacle3_y: {input_vector[15]:.2f}\n"
    text += f"Reward: {reward:.2f}\n"
    text += f"Episode: {episode}\n"
    text += f"Loss: {str(loss_item)}\n"
    return text

def choose_action(state, model):
    if np.random.rand() < EPSILON:
        # Случайное действие (эксплорейшн)
        return torch.FloatTensor(1, OUTPUT_DIM).uniform_(-1, 1)
    else:
        # Выбор действия на основе Q-значений (эксплуатация)
        with torch.no_grad():
            action = model(state)
            return action.view(1, -1)
        
def replay(memory, model, optimizer, loss_fn, batch_size):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)

    for state, action, reward, next_state, done in batch:
        state_tensor = state.clone().detach().unsqueeze(0)  # размер (1, state_size)
        next_state_tensor = next_state.clone().detach().unsqueeze(0) # размер (1, state_size)
        target = reward
        if not done:
            # Обновление Q-значения
            target += GAMMA * torch.max(model(next_state_tensor)).item()

        target_f = model(state_tensor).squeeze().clone()
        action_0 = action[0][0]
        action_1 = action[0][1]

        # Обновляем значения target_f с учетом action
        target_f[0] = action_0 * target  # Обновляем значение для первого выхода
        target_f[1] = action_1 * target  # Обновляем значение для второго выхода

        # Обучение модели
        optimizer.zero_grad()
        loss = loss_fn(target_f, model(state_tensor).squeeze())
        loss.backward()
        optimizer.step()

    return loss.item()

def soft_update(target_model, model, tau):
    """Обновление весов модели target по мере продвижения основной модели"""
    for target_param, local_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

grid = {}
def main():
    # Инициализация DQN модели, оптимизатора и функции потерь
    model = DQN(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM)
    # model.load_model("model10.pth")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    # Буфер реплея
    batch_size = 80  # Размер батча для обучения
    memory = deque(maxlen=batch_size * 20)  # Опыт для реплея

    organism = Organism()
    reward = Reward()
    matrix_plotter = MatrixPlotter(model.get_weights()[2])
    vectors_plotter = VectorsPlotter()
    statistics = Statistics()

    for episode in range(10000):  # Количество эпизодов
        if episode % 100 == 0:
            organism.reset_counters()
            obstacles = []
            for _ in range(OBSTACLE_QUANTITY):
                obstacles.append(Obstacle(organism, obstacles))
            foods = []
            for _ in range(FOOD_QUANTITY):
                foods.append(Food(obstacles, organism, foods))
            start_time = pygame.time.get_ticks()
        state = organism.get_input_vector(obstacles, foods)
        done = False
        reward.reset()
        memory.clear()
        loss_item = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    done = False

            grid.clear()
            for obstacle in obstacles:
                add_to_grid(obstacle)
            for food in foods:
                add_to_grid(food)
            
            screen.fill(WHITE)
            surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)

            for obstacle in obstacles:
                obstacle.move()
                obstacle.draw(screen)

            for food in foods:
                food.draw(screen)

            # Выбор действия
            next_state = organism.get_input_vector(obstacles, foods)
            action = choose_action(next_state.clone().detach().unsqueeze(0).requires_grad_(True), model)

            # for organism in organisms:
            done = organism.move(action, obstacles, reward, foods)
            reward_value = reward.get()
            pygame.draw.circle(screen, BLUE, (int(organism.x), int(organism.y)), ORGANISM_RADIUS)

            draw_dish()

            # Сохранение опыта в буфер
            memory.append((state, action, reward_value, next_state, done))

            # Обучение модели
            if len(memory) % batch_size == 0 or done:
                loss_item = replay(memory, model, optimizer, loss_fn, batch_size)
                if len(memory) == memory.maxlen:
                    memory.clear()

            state = next_state

            # Render input vector
            if organism.input_vector is not None:
                input_text = get_input_text(organism.input_vector, reward_value, episode, loss_item)
            render_text(screen, input_text, screen.get_width() - 400, 10)

            # Отрисовка векторов
            vectors_plotter.update(organism, reward_value)
            vectors_plotter.draw_vectors_plotter(screen)
            vectors_plotter.draw_vectors(organism, surface)
            if len(memory) % batch_size == 0 or done:
                vectors_plotter.reset()

            # Отриосвка потерь
            if loss_item != None and (len(memory) % batch_size == 0 or done):
                matrix_plotter.update_loss(loss_item)
            matrix_plotter.draw_loss(screen)

            # Отрисовка весов
            # matrix_plotter.update(model.get_weights()[2])
            # matrix_plotter.draw(screen)

            # Отрисовка наград
            reward.draw(screen)

            # Отрисовка счетчика эпизодов
            statistics.update_energy_max(organism.energy)
            statistics.update_total_reward(reward.get())
            statistics.total_time = (pygame.time.get_ticks() - start_time) / 1000
            if episode % 100 == 0:
                statistics.create_episode(episode // 100 + 1, organism)             
            statistics.update_episode(episode // 100 + 1, organism)
            statistics.draw_statistics(screen)

            if done and episode % 100 == 0:
                print(f"Episode: {episode}!")
                # Сохранение модели
                torch.save(model.state_dict(), f"model{episode // 100}.pth")

            reward.reset()

            screen.blit(surface, (0, 0))
            pygame.display.flip()
            # clock.tick(60)  # Ограничение FPS

if __name__ == "__main__":
    main()
    pygame.quit()
    sys.exit()