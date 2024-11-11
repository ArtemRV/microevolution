import pygame
import random
import numpy as np
from collections import OrderedDict
from classes_and_functions.colors import Colors

# font = pygame.font.SysFont(None, 18)
# colors = Colors()

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
        self.average_loss_log = 0
        self.average_loss_max = 0
        self.average_loss_max_log = 0

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
        # Среднее значение потерь
        self.average_loss = sum(self.loss) / len(self.loss)
        self.average_loss_log = self.log_value(self.average_loss)
        # Максимальное среднее значение потерь
        if self.average_loss > self.average_loss_max:
            self.average_loss_max = self.average_loss
            self.average_loss_max_log = self.log_value(self.average_loss_max)
    
    def log_value(self, value):
        if value > 1:
            return np.log(value) * 20
        else:
            return value

    def draw_loss(self, screen, colors, font):
        shift_x = 10
        plotter_shift_x = 30
        shift_y = screen.get_height() - 300
        for i, loss_item in enumerate(self.loss):
            loss_value = self.log_value(loss_item)
            pygame.draw.line(screen, colors.RED, (i + shift_x + plotter_shift_x, shift_y), (i + shift_x + plotter_shift_x, shift_y - loss_value), 1)
        # Добавить шкалу логарифмического масштаба
        render_text(screen, "0", shift_x, shift_y, colors, font)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y), (shift_x + plotter_shift_x + self.max_loss_count, shift_y), 1)
        render_text(screen, "10", shift_x, shift_y - 46, colors, font)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y - 46), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 46), 1)
        render_text(screen, "100", shift_x, shift_y - 92, colors, font)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y - 92), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 92), 1)
        render_text(screen, "1000", shift_x, shift_y - 138, colors, font)
        pygame.draw.line(screen, colors.BLACK, (shift_x + plotter_shift_x, shift_y - 138), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - 138), 1)
        # Отрисовка среднего значения
        render_text(screen, f"Av loss: {round(self.average_loss, 2)}", shift_x + plotter_shift_x + self.max_loss_count + 5, shift_y - self.average_loss_log, colors, font)
        render_text(screen, f"Max av loss: {round(self.average_loss_max, 2)}", shift_x + plotter_shift_x + self.max_loss_count + 5, shift_y - self.average_loss_max_log - 10, colors, font)
        pygame.draw.line(screen, colors.AZURE, (shift_x + plotter_shift_x, shift_y - self.average_loss_log), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - self.average_loss_log), 1)
        pygame.draw.line(screen, colors.BLUE, (shift_x + plotter_shift_x, shift_y - self.average_loss_max_log), (shift_x + plotter_shift_x + self.max_loss_count, shift_y - self.average_loss_max_log), 1)  

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

    def draw_vectors_plotter(self, screen, colors, font):
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
            render_text(screen, f"Rew: {round(self.rewards[i], 2)}", draw_center[0] + 20  + shift_x, draw_center[1] + shift_y, colors, font)

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

    def draw_statistics(self, settings, screen, colors, font):
        if settings['Statistics']:
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
                render_text(screen, f"Episode: {episode}", total_x_shift, shift_y, colors, font)
                
                for i, (label, key) in enumerate(data_keys):
                    value = round(data[key], 2) if isinstance(data[key], float) else data[key]
                    render_text(screen, f"{label}: {value}", total_x_shift, shift_y + (i + 1) * 10, colors, font)

    def manage_statistics_update(self, settings, episode, organism, start_time, reward):
        self.update_energy_max(organism.energy)
        self.update_total_reward(reward.get())
        self.total_time = (pygame.time.get_ticks() - start_time) / 1000
        if episode % settings['Episode length'] == 0:
            self.create_episode(episode // settings['Episode length'] + 1, organism)
        self.update_episode(episode // settings['Episode length'] + 1, organism)


def render_text(screen, text, x, y, colors, font):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        img = font.render(line, True, colors.BLACK)
        screen.blit(img, (x, y + i * 15))
