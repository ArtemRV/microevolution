import numpy as np
import random
import sys
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from classes_and_functions.objects import Grid, Dish, Organism, Obstacle, Food
from classes_and_functions.colors import Colors
from classes_and_functions.menu import menu
from classes_and_functions.button import load_button_settings
from classes_and_functions.charts import MatrixPlotter, VectorsPlotter, Statistics

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
EPISODES = 10000
EPISODE_LENGTH = 100

# Параметры обучения
EPSILON = 0.1  # Для epsilon-greedy стратегии
GAMMA = 0.99  # Фактор дисконтирования
TAU = 0.005  # Скорость копирования весов основной сети в target сеть

# Buttons
MENU = False
PAUSE = False


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


# Текстовый вывод
def render_text(screen, text, x, y):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        img = font.render(line, True, colors.BLACK)
        screen.blit(img, (x, y + i * 15))

def get_input_text(input_vector, reward, episode, loss_item):
    text = "Input vector:\n"
    text += f"food_x: {input_vector[0]:.2f}     food_y: {input_vector[1]:.2f}\n"
    text += f"food_x: {input_vector[2]:.2f}     food_y: {input_vector[3]:.2f}\n"
    text += f"food_x: {input_vector[4]:.2f}     food_y: {input_vector[5]:.2f}\n"
    text += f"vx: {input_vector[6]:.2f}         vy: {input_vector[7]:.2f}\n"
    text += f"distance_to_boundary: {input_vector[8]:.2f}\n"
    text += f"energy: {input_vector[9]:.2f}\n"
    text += f"obstacle1_x: {input_vector[10]:.2f}   obstacle1_y: {input_vector[11]:.2f}\n"
    text += f"obstacle1_vx: {input_vector[12]:.2f}  obstacle1_vy: {input_vector[13]:.2f}\n"
    text += f"obstacle2_x: {input_vector[14]:.2f}   obstacle2_y: {input_vector[15]:.2f}\n"
    text += f"obstacle2_vx: {input_vector[16]:.2f}  obstacle2_vy: {input_vector[17]:.2f}\n"
    text += f"obstacle3_x: {input_vector[18]:.2f}   obstacle3_y: {input_vector[19]:.2f}\n"
    text += f"obstacle3_vx: {input_vector[20]:.2f}  obstacle3_vy: {input_vector[21]:.2f}\n"
    text += f"Reward: {reward:.2f}\n"
    text += f"Episode: {episode}\n"
    text += f"Loss: {str(loss_item)}\n"
    return text

def choose_action(state, model):
    if np.random.rand() < EPSILON:
        # Случайное действие (эксплорейшн)
        return torch.FloatTensor(1, OUTPUT_DIM).uniform_(-1, 1)
    else:
        return model_action(state, model)
        
def model_action(state, model):
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

def manage_learning(settings, state, action, reward_value, next_state, done, model, optimizer, loss_fn, memory, batch_size, loss_item):
    # Сохранение опыта в буфер
    if settings['Learning']:
        memory.append((state, action, reward_value, next_state, done))

        # Обучение модели
        if len(memory) % batch_size == 0 or done:
            loss_item = replay(memory, model, optimizer, loss_fn, batch_size)
            if len(memory) == memory.maxlen:
                memory.clear()
    return loss_item

def manage_buttons(screen, colors, clock, settings, buttons):
    for button in buttons:
        button.update_rect(screen.get_width())
        button.check_hover()
        button.draw(screen)
    
    if MENU:
        settings = menu(screen, colors, clock, settings)
        menu_status()

def handle_events(buttons):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
            menu_status()
        if event.type == pygame.MOUSEBUTTONDOWN:
            for button in buttons:
                if button.is_clicked(event):
                    button.action()
                    break

def menu_status():
    global MENU
    MENU = not MENU

def pause():
    global PAUSE
    PAUSE = not PAUSE

ACTIONS = {
    "menu": menu_status,
    "pause": pause
}

grid = Grid(GRID_SIZE)

# Инициализация DQN модели, оптимизатора и функции потерь
model = DQN(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM)
# model.load_model("model20.pth")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Буфер реплея
batch_size = 80  # Размер батча для обучения
memory = deque(maxlen=batch_size * 20)  # Опыт для реплея

buttons = load_button_settings(colors, "settings/buttons.yml", ACTIONS, 'main_buttons')

def main():
    settings = menu(screen, colors, clock)
    settings['EPISODE_LENGTH'] = EPISODE_LENGTH
    dish = Dish(DISH_RADIUS, DISH_COLLISION_PENALTY, DISH_PENALTY_MAX, screen)
    organism = Organism(dish, ORGANISM_RADIUS, SPEED, COLOR, INITIAL_ENERGY, ENERGY_LOSS_RATE, SPEED_ENERGY_LOSS_MULTIPLIER, ENERGY_DEPLETION_PENALTY)
    reward = Reward()
    matrix_plotter = MatrixPlotter(model.get_weights()[2])
    vectors_plotter = VectorsPlotter()
    statistics = Statistics()

    for episode in range(EPISODES):
        if episode % settings['EPISODE_LENGTH'] == 0:
            organism.reset_counters()
            obstacles = [Obstacle(OBSTACLE_RADIUS, dish, organism, [], OBSTACLE_COLLISION_PENALTY, OBSTACLE_SPEED) for _ in range(OBSTACLE_QUANTITY)]
            foods = [Food(FOOD_RADIUS, FOOD_ENERGY, dish, organism, obstacles, [], FOOD_REWARD) for _ in range(FOOD_QUANTITY)]
            start_time = pygame.time.get_ticks()
            paused_time = 0
        state = organism.get_input_vector(dish, obstacles, foods)
        done = False
        reward.reset()
        memory.clear()
        loss_item = 0

        while not done:
            handle_events(buttons)
            
            if PAUSE:
                if paused_time == 0:
                    paused_time = pygame.time.get_ticks()

                manage_buttons(screen, colors, clock, settings, buttons)
            else:
                if paused_time != 0:
                    start_time += pygame.time.get_ticks() - paused_time
                    paused_time = 0

                screen.fill(colors.WHITE)
                surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
                
                manage_buttons(screen, colors, clock, settings, buttons)

                for obstacle in obstacles:
                    obstacle.move(dish, grid)

                # Выбор действия
                next_state = organism.get_input_vector(dish, obstacles, foods)
                if settings['Learning']:
                    action = choose_action(next_state.clone().detach().unsqueeze(0).requires_grad_(True), model)
                else:
                    action = model_action(next_state, model)

                # for organism in organisms:
                done = organism.move(action, dish, obstacles, reward, foods)

                if DRAW_ALL:
                    draw_all(screen, dish, obstacles, foods, organism)

                # Управление обучением
                loss_item = manage_learning(settings, state, action, reward.get(), next_state, done, model, optimizer, loss_fn, memory, batch_size, loss_item)

                state = next_state

                # Render input vector
                if settings['Input vector']:
                    if organism.input_vector is not None:
                        input_text = get_input_text(organism.input_vector, reward.get(), episode, loss_item)
                    render_text(screen, input_text, 10, 10)

                # Отрисовка векторов
                if settings['Vectors']:
                    vectors_plotter.update(organism, reward.get())
                    vectors_plotter.draw_vectors_plotter(screen, colors, font)
                    vectors_plotter.draw_vectors(organism, surface, dish)
                    if len(memory) % batch_size == 0 or done:
                        vectors_plotter.reset()

                # Отриосвка потерь
                if settings['Loss']:
                    if loss_item != None and (len(memory) % batch_size == 0 or done):
                        matrix_plotter.update_loss(loss_item)
                    matrix_plotter.draw_loss(screen, colors, font)

                # Отрисовка весов
                # matrix_plotter.update(model.get_weights()[2])
                # matrix_plotter.draw(screen)

                # Отрисовка наград
                if settings['Reward']:
                    reward.draw(screen)

                # Обновление и отрисовка статистики
                statistics.manage_statistics_update(settings, episode, organism, start_time, reward)
                statistics.draw_statistics(settings, screen, colors, font)

                if done and episode % settings['EPISODE_LENGTH'] == 0:
                    print(f"Episode: {episode}!")
                    # Сохранение модели
                    torch.save(model.state_dict(), f"model{episode // settings['EPISODE_LENGTH']}.pth")

                reward.reset()

                screen.blit(surface, (0, 0))
            pygame.display.flip()
            # clock.tick(60)  # Ограничение FPS

if __name__ == "__main__":
    main()
    pygame.quit()
    sys.exit()