import sys
import yaml
import pygame
import threading
import queue
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from classes_and_functions.model import DQN
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

# Settings
settings_file = "settings/settings.yml"

# Размер ячеек сетки
GRID_SIZE = 50

# Параметры нейронной сети
INPUT_DIM = 22
HIDDEN_DIMS = [64, 32]
OUTPUT_DIM = 2

# Buttons
MENU = False
PAUSE = False


# Определение архитектуры нейронной сети для DQN



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
    organism.draw(screen, dish)
    dish.draw(screen, colors)

# def thread_learning(**kwargs):
#     dish = kwargs['dish']
#     organism = kwargs['organism']
#     obstacles = kwargs['obstacles']
#     foods = kwargs['foods']
#     model = kwargs['model']
#     settings = kwargs['settings']
#     state = kwargs['state']

#     reward = Reward()
#     batch_size = settings['BATCH_SIZE']
#     memory = deque(maxlen=batch_size * 20)
#     loss_fn = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])

#     # Move obstacles
#     if obstacles is not []:
#         for obstacle in obstacles:
#             obstacle.move(dish, grid)

#     # Выбор действия
#     next_state = organism.get_input_vector(dish, obstacles, foods)
#     action = choose_action(next_state.clone().detach().unsqueeze(0).requires_grad_(True), model, settings)

#     # for organism in organisms:
#     done = organism.move(action, dish, obstacles, reward, foods, settings)

#     # Управление обучением
#     loss_item = manage_learning(settings, state, action, reward.get(), next_state, done, model, optimizer, loss_fn, memory, batch_size, loss_item)

#     state = next_state

# def train_network(status_queue, **kwargs):
#     settings = kwargs['settings']
#     for episode in range(settings['Number of episodes']):
#         if episode % settings['Episode length'] == 0:
#             organism.reset_counters()
#             obstacles = [Obstacle(dish, organism, [], settings) for _ in range(settings['OBSTACLE_QUANTITY'])]
#             foods = [Food(dish, organism, obstacles, [], settings) for _ in range(settings['FOOD_QUANTITY'])]
#             start_time = pygame.time.get_ticks()
#             paused_time = 0
#         state = organism.get_input_vector(dish, obstacles, foods)
#         done = False
#         reward.reset()
#         memory.clear()
#         loss_item = 0

#         while not done:
#             pass

# # Функция для запуска обучения в отдельном потоке
# def start_training(num_epochs, status_queue):
#     training_thread = threading.Thread(target=train_network, args=(status_queue, num_epochs))
#     training_thread.start()

def manage_learning(settings, state, action, reward_value, next_state, done, model, optimizer, loss_fn, memory, batch_size, loss_item):
    # Сохранение опыта в буфер
    if settings['Learning']:
        memory.append((state, action, reward_value, next_state, done))

        # Обучение модели
        if len(memory) % batch_size == 0 or done:
            loss_item = model.replay(memory, optimizer, loss_fn, batch_size, settings)
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

def upload_settings(settings, settings_file):
    with open(settings_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        settings.update(data)

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

buttons = load_button_settings(colors, "settings/buttons.yml", ACTIONS, 'main_buttons')
grid = Grid(GRID_SIZE)

def main():
    settings = menu(screen, colors, clock)
    upload_settings(settings, settings_file)

    # Инициализация DQN модели, оптимизатора и функции потерь
    model = DQN(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM)
    model.load_model(settings)
    optimizer = optim.Adam(model.parameters(), lr=settings['LEARNING_RATE'])
    loss_fn = nn.MSELoss()

    status_queue = queue.Queue()

    # Буфер реплея
    batch_size = settings['BATCH_SIZE']  # Размер батча для обучения
    memory = deque(maxlen=batch_size * 20)  # Опыт для реплея

    dish = Dish(settings, screen)
    organism = Organism(dish, settings, colors)
    reward = Reward()
    matrix_plotter = MatrixPlotter(model.get_weights()[2])
    vectors_plotter = VectorsPlotter()
    statistics = Statistics()

    for episode in range(settings['Number of episodes']):
        if episode % settings['Episode length'] == 0:
            organism.reset_counters()
            obstacles = [Obstacle(dish, organism, [], settings) for _ in range(settings['OBSTACLE_QUANTITY'])]
            foods = [Food(dish, organism, obstacles, [], settings) for _ in range(settings['FOOD_QUANTITY'])]
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
                    action = model.choose_action(next_state.clone().detach().unsqueeze(0).requires_grad_(True), settings, OUTPUT_DIM)
                else:
                    action = model.model_action(next_state)

                # for organism in organisms:
                done = organism.move(action, dish, obstacles, reward, foods, settings)

                if settings['Draw all']:
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

                model.save_model(settings, episode, done)

                reward.reset()

                screen.blit(surface, (0, 0))
            pygame.display.flip()
            # clock.tick(60)  # Ограничение FPS

if __name__ == "__main__":
    main()
    pygame.quit()
    sys.exit()