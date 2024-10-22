import numpy as np
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

pygame.init()
torch.autograd.set_detect_anomaly(True)

# Параметры окна
WIDTH, HEIGHT = 1400, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Petri Dish Simulation")

# Основные цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Радиус чашки Петри и препятствий
RADIUS_DISH = 350
OBSTACLE_RADIUS = 10
ORGANISM_RADIUS = 15
FOOD_RADIUS = 5

OBSTACLE_QUANTITY = 30
FOOD_QUANTITY = 60

# Центр чашки Петри
CENTER = (WIDTH // 2, HEIGHT // 2)

INPUT_DIM = 16
OUTPUT_DIM = 2
HIDDEN_DIM1 = 64
HIDDEN_DIM2 = 32
HIDDEN_DIM3 = 16
LEARNING_RATE = 0.0001

# Параметры обучения
EPSILON = 0.1  # Для epsilon-greedy стратегии
GAMMA = 0.99  # Фактор дисконтирования
TAU = 0.005  # Скорость копирования весов основной сети в target сеть

# Font for rendering text
font = pygame.font.SysFont(None, 18)
clock = pygame.time.Clock()

# Класс для препятствий
class Obstacle:
    def __init__(self):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, RADIUS_DISH - OBSTACLE_RADIUS)
        self.x = CENTER[0] + radius * np.cos(angle)
        self.y = CENTER[1] + radius * np.sin(angle)
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-2, 2)

    def move(self):
        # Проверка столкновений с чашкой Петри и другими препятствиями
        if check_dish_collision(self.x + self.vx, self.y + self.vy, OBSTACLE_RADIUS):
            angle_rebound = calculata_rebound_angle(self.x, self.y, CENTER[0], CENTER[1], self.vx, self.vy)
            self.vx = -np.cos(angle_rebound) * np.hypot(self.vx, self.vy)
            self.vy = -np.sin(angle_rebound) * np.hypot(self.vx, self.vy)
        for obstacle in obstacles:
            if obstacle == self:
                continue
            if check_collision(self.x + self.vx, self.y + self.vy, OBSTACLE_RADIUS, obstacle.x, obstacle.y, OBSTACLE_RADIUS):
                angle_rebound = calculata_rebound_angle(self.x, self.y, obstacle.x, obstacle.y, self.vx, self.vy)
                self.vx = -np.cos(angle_rebound) * np.hypot(self.vx, self.vy)
                self.vy = -np.sin(angle_rebound) * np.hypot(self.vx, self.vy)
                angle_rebound = calculata_rebound_angle(obstacle.x, obstacle.y, self.x, self.y, obstacle.vx, obstacle.vy)
                obstacle.vx = -np.cos(angle_rebound) * np.hypot(obstacle.vx, obstacle.vy)
                obstacle.vy = -np.sin(angle_rebound) * np.hypot(obstacle.vx, obstacle.vy)
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), OBSTACLE_RADIUS)

# Класс для пищи
class Food:
    def __init__(self):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, RADIUS_DISH - FOOD_RADIUS)
        self.x = CENTER[0] + distance * np.cos(angle)
        self.y = CENTER[1] + distance * np.sin(angle)

    def draw(self, screen):
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), FOOD_RADIUS)

    def check_eaten(self, organism):
        # Проверка столкновения между организмом и едой
        distance = np.hypot(self.x - organism.x, self.y - organism.y)
        return distance < (FOOD_RADIUS + ORGANISM_RADIUS)

    def regenerate(self):
        """Регенерация пищи в новом месте."""
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, RADIUS_DISH - FOOD_RADIUS)
        self.x = CENTER[0] + radius * np.cos(angle)
        self.y = CENTER[1] + radius * np.sin(angle)


class MatrixPlotter:
    def __init__(self, matrix):
        self.matrix = matrix
        width, height = self.matrix.shape
        self.width = width
        self.height = height
        self.colors = self.generate_colors()
        self.loss = []
        self.loss_max = 200

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
        if len(self.loss) > self.loss_max:
            self.loss.pop(0)
        self.loss.append(loss)

    def draw_loss(self, screen):
        shift_x = 10
        shift_y = screen.get_height() - 200
        for i, loss_item in enumerate(self.loss):
            pygame.draw.line(screen, RED, (i + shift_x, shift_y), (i + shift_x, shift_y - loss_item), 2)


class VectorsPlotter:
    def __init__(self):
        self.vectors = []
        self.start = []
        self.draw_center = [screen.get_width() - 300, screen.get_height() // 2 - 100]
        self.rewards = []

    def update(self, organism, reward):
        self.vectors.append([organism.vx, organism.vy])
        if len(self.start) == 0:
            self.start = [organism.x, organism.y]
        self.rewards.append(reward)

    def draw(self, screen):
        a = 20
        x = 100
        for i, vector in enumerate(self.vectors):
            shift_x = i // a * x
            shift_y = i * a - (i // a) * a * a
            start_pos = [self.draw_center[0] + shift_x, self.draw_center[1] + shift_y]
            end_pos = [self.draw_center[0] + int(vector[0] * 10)  + shift_x, self.draw_center[1] + int(vector[1] * 10) + shift_y]
            pygame.draw.line(screen, RED, start_pos, end_pos, 2)  # Толщина линии фиксирована на 2 пикселя
            pygame.draw.circle(screen, BLUE, end_pos, 5)
            render_text(screen, f"Rew: {round(self.rewards[i], 2)}", self.draw_center[0] + 20  + shift_x, self.draw_center[1] + shift_y)

    def reset(self):
        self.vectors = []
        self.start = []
        self.rewards = []


# Определение архитектуры нейронной сети для DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM1)  # Входной слой (12 параметров)
        self.fc2 = nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)  # Скрытый слой
        self.fc3 = nn.Linear(HIDDEN_DIM2, HIDDEN_DIM3)  # Скрытый слой
        self.fc4 = nn.Linear(HIDDEN_DIM3, OUTPUT_DIM)   # Выходной слой (две координаты для перемещения)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
    
    def get_weights(self):
        weights_fc1 = self.fc1.weight.data.numpy().squeeze()  # Матрица весов для первого слоя
        weights_fc2 = self.fc2.weight.data.numpy().squeeze()  # Матрица весов для второго слоя
        weights_fc3 = self.fc3.weight.data.numpy().squeeze()  # Матрица весов для третьего слоя
        weights_fc4 = self.fc4.weight.data.numpy().squeeze()  # Матрица весов для выходного слоя
        return weights_fc1, weights_fc2, weights_fc3, weights_fc4
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))


# Класс для организма
class Organism:
    def __init__(self):
        self.x = CENTER[0]
        self.y = CENTER[1]
        self.speed = 1
        self.vx = np.random.uniform(-self.speed, self.speed)
        self.vy = np.random.uniform(-self.speed, self.speed)
        self.energy = 10
        self.input_vector = None
        self.reset_status = False
        self.done = False

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
            np.hypot(self.x - CENTER[0], self.y - CENTER[1]) + ORGANISM_RADIUS - RADIUS_DISH,
            self.energy,
            self.x - closest_obstacles[0].x, self.y - closest_obstacles[0].y,
            self.x - closest_obstacles[1].x, self.y - closest_obstacles[1].y,
            self.x - closest_obstacles[2].x, self.y - closest_obstacles[2].y
        ])
        return torch.tensor(self.input_vector, dtype=torch.float32)

    def move(self, action, obstacles, reward):
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
        self.food_collision(reward)
        self.energy_update(reward)

        if self.reset_status == True:
            self.reset(obstacles)

        return self.done

    def food_collision(self, reward):
        for i, food in enumerate(self.closest_food):
            if food.check_eaten(self):
                self.energy += 5
                foods.remove(food)
                foods.append(Food())
                new_reward = 30
                self.done = True
            else:
                new_reward = (3 - i) / 3 * calculate_reward(self.x, self.y, self.vx, self.vy, food.x, food.y, 300)
            reward.update(new_reward, 'eat')

    def obstacle_collision(self, reward):
        for i, obstacle in enumerate(self.closest_obstacles):
            if check_collision(self.x, self.y, ORGANISM_RADIUS, obstacle.x, obstacle.y, OBSTACLE_RADIUS):
                new_reward = -50
                self.reset_status = True
            else:
                new_reward = (3 - i) / 3 * calculate_obstacle_collision_reward(self.x, self.y, obstacle.x, obstacle.y, -40)
            reward.update(new_reward, 'obstacle_collision')

    def energy_update(self, reward):
        self.energy -= 0.01 + np.hypot(self.vx, self.vy) * 0.001  # Штраф за скорость
        if self.energy <= 0:
            new_reward = -40
            self.reset_status = True
        else:
            new_reward = calculate_simple_reward(self.energy, -3)
        reward.update(new_reward, 'energy')
    
    def dish_collision(self, reward):
        overlap = calculate_distance(self.x, self.y, CENTER[0], CENTER[1]) + ORGANISM_RADIUS - RADIUS_DISH
        if overlap > 0:
            new_reward = -30
            self.reset_status = True
        elif overlap > -30:
            new_reward = calculate_simple_reward(overlap, -15)
        else:
            new_reward = 0
        reward.update(new_reward, 'dish_collision')


    def reset(self, obstacles):
        """Сброс позиции организма при столкновении с препятствием."""
        while True:
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, RADIUS_DISH - ORGANISM_RADIUS)
            self.x = CENTER[0] + radius * np.cos(angle)
            self.y = CENTER[1] + radius * np.sin(angle)
            collision = False
            for obstacle in obstacles:
                if check_collision(self.x, self.y, ORGANISM_RADIUS, obstacle.x, obstacle.y, OBSTACLE_RADIUS):
                    collision = True
                    break
            if not collision:
                break
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-2, 2)
        self.energy = 10
        self.done = True


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
        shift_y = screen.get_height() - 100
        render_text(screen, f"Eat reward:                   {round(self.eat_reward, 2)}", shift_x, shift_y)
        render_text(screen, f"Obstacle collision reward:    {round(self.obstacle_collision_reward, 2)}", shift_x, shift_y + 10)
        render_text(screen, f"Dish collision reward:        {round(self.dish_collision_reward, 2)}", shift_x, shift_y + 20)
        render_text(screen, f"Energy reward:                {round(self.energy_reward, 2)}", shift_x, shift_y + 30)

    def get(self):
        return self.reward

    def reset(self):
        self.reward = 0
        self.eat_reward = 0
        self.obstacle_collision_reward = 0
        self.dish_collision_reward = 0
        self.energy_reward = 0
        

def calculate_simple_reward(parameter, factor):
    calculated_reward = -5 / abs(parameter)
    if calculated_reward < factor:
        calculated_reward = factor
    return calculated_reward

def calculate_reward(x1, y1, vx, vy, x2, y2, factor):
    initial_distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
    new_distance = np.linalg.norm(np.array([x1 + vx, y1 + vy]) - np.array([x2, y2]))
    distance_diff = initial_distance - new_distance
    calculated_reward = distance_diff * factor * (1 / new_distance)
    return calculated_reward

def calculate_obstacle_collision_reward(target_x, target_y, current_x, current_y, limit):
    distance = np.hypot(target_x - current_x, target_y - current_y) - ORGANISM_RADIUS - OBSTACLE_RADIUS
    reward = (-10 / distance) * 30 + 7
    if reward < limit:
        reward = limit
    if reward > 0:
        reward = 0
    return reward

# Функции для рисования и физики
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

        # Если action[0] — тензор размерности [2], нужно извлечь скаляр из каждого элемента
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


# Инициализация DQN модели, оптимизатора и функции потерь
model = DQN()
# model.load_model("60x400.pth")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Буфер реплея
batch_size = 40  # Размер батча для обучения
memory = deque(maxlen=batch_size * 10)  # Опыт для реплея

obstacles = [Obstacle() for _ in range(OBSTACLE_QUANTITY)]
foods = [Food() for _ in range(FOOD_QUANTITY)]
organism = Organism()
reward = Reward()
matrix_plotter = MatrixPlotter(model.fc1.weight.data.numpy().squeeze())
vectors_plotter = VectorsPlotter()

# Процесс обучения
for episode in range(1000):  # Количество эпизодов
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

        screen.fill(WHITE)
        for obstacle in obstacles:
            obstacle.move()
            obstacle.draw(screen)

        for food in foods:
            food.draw(screen)

        # Выбор действия
        next_state = organism.get_input_vector(obstacles, foods)
        action = choose_action(next_state.clone().detach().unsqueeze(0).requires_grad_(True), model)

        # for organism in organisms:
        done = organism.move(action, obstacles, reward)
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
        render_text(screen, input_text, WIDTH - 300, 20)

        # Отрисовка векторов
        vectors_plotter.update(organism, reward_value)
        vectors_plotter.draw(screen)
        if len(memory) % batch_size == 0 or done:
            vectors_plotter.reset()

        # Отриосвка потерь
        if loss_item != None and (len(memory) % batch_size == 0 or done):
            matrix_plotter.update_loss(loss_item)
        matrix_plotter.draw_loss(screen)

        # Отрисовка весов
        matrix_plotter.update(model.fc1.weight.data.numpy().squeeze())
        matrix_plotter.draw(screen)

        # Отрисовка наград
        reward.draw(screen)

        if done and episode % 100 == 0:
            print(f"Episode: {episode}!")
            # Сохранение модели
            torch.save(model.state_dict(), "model.pth")
            
            # Сохранение весов в файл txt
            weights_fc1, weights_fc2, weights_fc3, weights_fc4 = model.get_weights()
            np.savetxt("weights_fc1.txt", weights_fc1, fmt='%.2f')
            np.savetxt("weights_fc2.txt", weights_fc2, fmt='%.2f')
            np.savetxt("weights_fc3.txt", weights_fc3, fmt='%.2f')
            np.savetxt("weights_fc4.txt", weights_fc4, fmt='%.2f')

        reward.reset()

        pygame.display.flip()
        # clock.tick(60)  # Ограничение FPS