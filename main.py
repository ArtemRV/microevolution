import numpy as np
import random
import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, OrderedDict
from classes_and_functions.objects import Grid, Dish, Organism, Obstacle, Food
from classes_and_functions.colors import Colors

pygame.init()
torch.autograd.set_detect_anomaly(True)

# Параметры окна
WIDTH, HEIGHT = 1500, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
# screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Micro evolution")

# Font for rendering text
font = pygame.font.SysFont(None, 18)
clock = pygame.time.Clock()

# Colors
colors = Colors()

# Buttons
DRAW_ALL = True

# Parameters for the dish
DISH_RADIUS = 350
DISH_COLLISION_PENALTY = -15
DISH_PENALTY_MAX = -30

# Obstacle parameters
OBSTACLE_RADIUS = 10
OBSTACLE_COLLISION_PENALTY = -30
OBSTACLE_SPEED = 1

# Food parameters
FOOD_RADIUS = 5
FOOD_ENERGY = 5
FOOD_REWARD = 20

# Organism parameters
ORGANISM_RADIUS = 15
SPEED = 1.5
INITIAL_ENERGY = 10
ENERGY_LOSS_RATE = 0.02
SPEED_ENERGY_LOSS_MULTIPLIER = 0.001
ENERGY_DEPLETION_PENALTY = -10
COLOR = colors.BLUE
COLOR2 = colors.ORANGE

# Размер ячеек сетки
GRID_SIZE = 50

# Количество препятствий и еды
OBSTACLE_QUANTITY = 30
FOOD_QUANTITY = 60

# Параметры нейронной сети
INPUT_DIM = 22
HIDDEN_DIMS = [64, 32]
OUTPUT_DIM = 2

LEARNING_RATE = 0.0001

# Параметры обучения
EPSILON = 0.1  # Для epsilon-greedy стратегии
GAMMA = 0.99  # Фактор дисконтирования
TAU = 0.005  # Скорость копирования весов основной сети в target сеть
        

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
            pygame.draw.line(screen, colors.RED, (i + shift_x + plotter_shift_x, shift_y), (i + shift_x + plotter_shift_x, shift_y - loss_value), 1)
        # Добавить шкалу логарифмического масштаба
        render_text(screen, "0", shift_x, shift_y)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y), (shift_x + plotter_shift_x + self.max_loss_count, shift_y), 1)
        render_text(screen, "10", shift_x, shift_y - 46)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y - 46), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 46), 1)
        render_text(screen, "100", shift_x, shift_y - 92)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y - 92), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 92), 1)
        render_text(screen, "1000", shift_x, shift_y - 138)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y - 138), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 138), 1)
        # Отрисовка среднего значения
        render_text(screen, f"Av loss: {round(self.average_loss, 2)}", shift_x + plotter_shift_x + self.max_loss_count + 5, shift_y - self.average_loss)
        render_text(screen, f"Max av loss: {round(self.average_loss_max, 2)}", shift_x + plotter_shift_x + self.max_loss_count + 5, shift_y - self.average_loss_max - 10)
        pygame.draw.line(screen, colors.AZURE, (shift_x + plotter_shift_x, shift_y - self.average_loss), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - self.average_loss), 1)
        pygame.draw.line(screen, colors.BLUE, (shift_x + plotter_shift_x, shift_y - self.average_loss_max), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - self.average_loss_max), 1)  


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
            pygame.draw.line(screen, colors.RED, start_pos, end_pos, 2)
            pygame.draw.circle(screen, colors.BLUE, end_pos, 5)
            render_text(screen, f"Rew: {round(self.rewards[i], 2)}", draw_center[0] + 20  + shift_x, draw_center[1] + shift_y)

    def reset(self):
        self.vectors = []
        self.start = []
        self.rewards = []

    def draw_vectors(self, organism, surface, dish):
        shift = organism.radius + 3
        reward_vectors = []

        # Предварительные вычисления для всех направлений
        for alfa in range(0, 360, 6):
            reward = 0
            radian_alfa = np.radians(alfa)
            vx = np.cos(radian_alfa)
            vy = np.sin(radian_alfa)
            
            # Подсчет награды от пищи и препятствий
            for i, food in enumerate(organism.closest_food):
                reward += organism.food_reward(food, i, organism.old_x + vx, organism.old_y + vy, vx, vy)
            for i, obstacle in enumerate(organism.closest_obstacles):
                reward += organism.obstacle_reward(obstacle, i, organism.old_x + vx, organism.old_y + vy, vx, vy)
                    
            # dish reward
            overlap = organism.dish_overlap(vx, vy, dish)
            reward += organism.dish_reward(overlap, dish)
            
            # Определение цвета и направления
            color = (0, 255, 0, 127) if reward >= 0 else (255, 0, 0, 127)
            reward_vectors.append((vx * abs(reward), vy * abs(reward), color))

        # Рисование всех векторов
        for idx, (vx, vy, color) in enumerate(reward_vectors):
            radian_alfa = np.radians(idx * 6)
            x = organism.x + shift * np.cos(radian_alfa)
            y = organism.y + shift * np.sin(radian_alfa)
            shift_x = x + dish.x
            shift_y = y + dish.y
            pygame.draw.line(surface, color, (shift_x, shift_y), (shift_x + vx, shift_y + vy), 1)


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


# Текстовый вывод
def render_text(screen, text, x, y):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        img = font.render(line, True, colors.BLACK)
        screen.blit(img, (x, y + i * 15))

def get_input_text(input_vector, reward, episode, loss_item):
    text = "Input vector:\n"
    text += f"food_x: {input_vector[0]:.2f} food_y: {input_vector[1]:.2f}\n"
    text += f"food_x: {input_vector[2]:.2f} food_y: {input_vector[3]:.2f}\n"
    text += f"food_x: {input_vector[4]:.2f} food_y: {input_vector[5]:.2f}\n"
    text += f"vx: {input_vector[6]:.2f} vy: {input_vector[7]:.2f}\n"
    text += f"distance_to_boundary: {input_vector[8]:.2f}\n"
    text += f"energy: {input_vector[9]:.2f}\n"
    text += f"obstacle1_x: {input_vector[10]:.2f} obstacle1_y: {input_vector[11]:.2f}\n"
    text += f"obstacle1_vx: {input_vector[12]:.2f} obstacle1_vy: {input_vector[13]:.2f}\n"
    text += f"obstacle2_x: {input_vector[14]:.2f} obstacle2_y: {input_vector[15]:.2f}\n"
    text += f"obstacle2_vx: {input_vector[16]:.2f} obstacle2_vy: {input_vector[17]:.2f}\n"
    text += f"obstacle3_x: {input_vector[18]:.2f} obstacle3_y: {input_vector[19]:.2f}\n"
    text += f"obstacle3_vx: {input_vector[20]:.2f} obstacle3_vy: {input_vector[21]:.2f}\n"
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

def draw_all(screen, dish, obstacles, foods, organism):
    dish.get_center(screen)
    grid.grid.clear()
    for obstacle in obstacles:
        grid.add_to_grid(obstacle)
    for food in foods:
        grid.add_to_grid(food)
        
    for obstacle in obstacles:
        obstacle.draw(screen, colors, dish)
    for food in foods:
        food.draw(screen, colors, dish)
    organism.draw(screen, colors, dish)
    dish.draw(screen, colors)

grid = Grid(GRID_SIZE)

# Инициализация DQN модели, оптимизатора и функции потерь
model = DQN(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM)
# model.load_model("model20.pth")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Буфер реплея
batch_size = 80  # Размер батча для обучения
memory = deque(maxlen=batch_size * 20)  # Опыт для реплея

def main():
    dish = Dish(DISH_RADIUS, DISH_COLLISION_PENALTY, DISH_PENALTY_MAX, screen)
    organism = Organism(dish, ORGANISM_RADIUS, SPEED, COLOR, INITIAL_ENERGY, ENERGY_LOSS_RATE, SPEED_ENERGY_LOSS_MULTIPLIER, ENERGY_DEPLETION_PENALTY)
    reward = Reward()
    matrix_plotter = MatrixPlotter(model.get_weights()[2])
    vectors_plotter = VectorsPlotter()
    statistics = Statistics()

    for episode in range(10000):  # Количество эпизодов
        if episode % 100 == 0:
            organism.reset_counters()
            obstacles = []
            for _ in range(OBSTACLE_QUANTITY):
                obstacles.append(Obstacle(OBSTACLE_RADIUS, dish, organism, obstacles, OBSTACLE_COLLISION_PENALTY, OBSTACLE_SPEED))
            foods = []
            for _ in range(FOOD_QUANTITY):
                foods.append(Food(FOOD_RADIUS, FOOD_ENERGY, dish, organism, obstacles, foods, FOOD_REWARD))
            start_time = pygame.time.get_ticks()
        state = organism.get_input_vector(dish, obstacles, foods)
        done = False
        reward.reset()
        memory.clear()
        loss_item = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    done = False
            
            screen.fill(colors.WHITE)
            surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)

            for obstacle in obstacles:
                obstacle.move(dish, grid)

            # Выбор действия
            next_state = organism.get_input_vector(dish, obstacles, foods)
            action = choose_action(next_state.clone().detach().unsqueeze(0).requires_grad_(True), model)

            # for organism in organisms:
            done = organism.move(action, dish, obstacles, reward, foods)

            reward_value = reward.get()

            if DRAW_ALL:
                draw_all(screen, dish, obstacles, foods, organism)

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
            render_text(screen, input_text, 10, 10)

            # Отрисовка векторов
            vectors_plotter.update(organism, reward_value)
            vectors_plotter.draw_vectors_plotter(screen)
            vectors_plotter.draw_vectors(organism, surface, dish)
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