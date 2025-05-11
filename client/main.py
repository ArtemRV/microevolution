import pygame
import numpy as np
from common.settings import Colors
from common.game_object import Environment, Organism, Food, Obstacle, Reward, Grid

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont(None, 24)

colors = Colors()

class RenderableOrganism(Organism):
    def draw(self, screen, colors, scale=1.0):
        pos = (self.pos * scale).astype(int)
        radius = int(self.radius * scale)
        pygame.draw.circle(screen, colors.BLUE, pos, radius)

    def move(self, action, foods, obstacles):
        acceleration = np.array(action) * self.settings['organism']['max_acceleration']
        self.vel += acceleration
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed and speed > 0:
            self.vel = self.vel / speed * self.max_speed
        prev_pos = self.pos.copy()
        self.pos += self.vel
        self.prev_action = action

        if not self.settings['rewards_enabled']:
            return False

        reward_settings = self.settings.get('rewards', {})
        self.steps_without_food += 1  # Увеличиваем счетчик шагов без еды

        """Approach Reward"""
        if reward_settings['approach']['enabled']:
            nearby_foods = [obj for obj in self.env.grid.get_nearby_objects(self.pos) if isinstance(obj, (Food, RenderableFood))]
            if nearby_foods:
                closest_food = min(nearby_foods, key=lambda f: np.linalg.norm(self.pos - f.pos))
                prev_dist = np.linalg.norm(prev_pos - closest_food.pos)
                curr_dist = np.linalg.norm(self.pos - closest_food.pos)
                if curr_dist < prev_dist:
                    self.env.reward.update(reward_settings['approach']['value'], 'approach')

        """Collision with the border"""
        dist_to_center = np.linalg.norm(self.pos - self.env.dish_center)
        if dist_to_center > self.env.dish_radius - self.radius and reward_settings['dish_collision']['enabled']:
            self.env.reward.update(reward_settings['dish_collision']['value'], 'dish_collision')
            if reward_settings['dish_collision']['end_episode']:
                return True

        """Checking nearby objects"""
        nearby_objects = self.env.grid.get_nearby_objects(self.pos)
        nearby_foods = [obj for obj in nearby_objects if isinstance(obj, (Food, RenderableFood))]
        nearby_obstacles = [obj for obj in nearby_objects if isinstance(obj, (Obstacle, RenderableObstacle))]

        # Награда за еду и сброс штрафа
        ate_food = False
        if reward_settings['eat']['enabled']:
            for food in nearby_foods[:]:
                if np.linalg.norm(self.pos - food.pos) < self.radius + food.radius:
                    self.env.reward.update(reward_settings['eat']['value'], 'eat')
                    self.energy += self.settings['organism']['energy_per_food']
                    self.food_eaten += 1
                    ate_food = True
                    if food in self.env.foods:
                        self.env.foods.remove(food)
                    new_food = RenderableFood(self.env, self.settings, [self] + self.env.obstacles + self.env.foods)
                    self.env.foods.append(new_food)

        # Сброс штрафа за энергию при поедании
        if ate_food:
            self.steps_without_food = 0
            self.current_energy_penalty = reward_settings['energy']['value']

        # Нарастающий штраф за энергию
        if reward_settings['energy']['enabled']:
            increment_steps = reward_settings['energy']['increment_steps']
            increment = reward_settings['energy']['increment']
            if self.steps_without_food >= increment_steps and self.steps_without_food % increment_steps == 0:
                self.current_energy_penalty -= increment  # Увеличиваем штраф (делаем более отрицательным)
            self.energy -= self.settings['organism']['energy_per_step']
            self.env.reward.update(self.current_energy_penalty, 'energy')
            if self.energy <= 0:
                return True

        # Столкновение с препятствием
        if reward_settings['obstacle_collision']['enabled']:
            for obstacle in nearby_obstacles:
                if np.linalg.norm(self.pos - obstacle.pos) < self.radius + obstacle.radius:
                    self.env.reward.update(reward_settings['obstacle_collision']['value'], 'obstacle_collision')
                    if reward_settings['obstacle_collision']['end_episode']:
                        return True

        # Награда за выживание
        if reward_settings['survival']['enabled'] and self.energy > 0:
            self.env.reward.update(reward_settings['survival']['value'], 'survival')

        return False

class RenderableFood(Food):
    def draw(self, screen, colors, scale=1.0):
        pos = (self.pos * scale).astype(int)
        radius = int(self.radius * scale)
        pygame.draw.circle(screen, colors.GREEN, pos, radius)

class RenderableObstacle(Obstacle):
    def draw(self, screen, colors, scale=1.0):
        pos = (self.pos * scale).astype(int)
        radius = int(self.radius * scale)
        pygame.draw.circle(screen, colors.RED, pos, radius)

class RenderableReward(Reward):
    def draw(self, screen, scale=1.0):
        y = int(10 * scale)
        texts = [
            f"Total Reward: {self.total:.2f}",
            f"Eat Reward: {self.eat:.2f}",
            f"Obstacle Collision: {self.obstacle_collision:.2f}",
            f"Dish Collision: {self.dish_collision:.2f}",
            f"Energy: {self.energy:.2f}",
            f"Approach: {self.approach:.2f}"
        ]
        for i, text in enumerate(texts):
            img = font.render(text, True, colors.BLACK)
            scaled_img = pygame.transform.scale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
            screen.blit(scaled_img, (int(10 * scale), y + int(i * 20 * scale)))

class RenderableEnvironment(Environment):
    def __init__(self, settings, screen, episode=0):
        self.screen = screen
        self.episode = episode
        self.settings = settings
        self.width = settings['general']['width']
        self.height = settings['general']['height']
        self.dish_radius = settings['general']['dish_radius']
        self.dish_center = (self.width // 2, 10 + settings['general']['dish_radius'])
        self.grid_size = settings['general']['grid_size']
        self.grid = Grid(self.grid_size, self.width, self.height)
        self.scale = 1.0
        
        # Use rendering-specific classes
        obstacle_quantity = min(settings['obstacle']['quantity'], 2 + episode // 200)
        self.agent = RenderableOrganism(self, settings)
        self.foods = []
        existing_objects = [self.agent]
        for _ in range(settings['food']['quantity']):
            food = RenderableFood(self, settings, existing_objects)
            self.foods.append(food)
            existing_objects.append(food)
        self.obstacles = []
        for _ in range(obstacle_quantity):
            obstacle = RenderableObstacle(self, settings, existing_objects)
            self.obstacles.append(obstacle)
            existing_objects.append(obstacle)
        self.reward = RenderableReward()
        self.update_grid()

    def reset(self):
        self.agent = RenderableOrganism(self, self.settings)
        self.foods = []
        existing_objects = [self.agent]
        for _ in range(self.settings['food']['quantity']):
            food = RenderableFood(self, self.settings, existing_objects)
            self.foods.append(food)
            existing_objects.append(food)
        self.obstacles = []
        obstacle_quantity = min(self.settings['obstacle']['quantity'], 2 + self.episode // 200)
        for _ in range(obstacle_quantity):
            obstacle = RenderableObstacle(self, self.settings, existing_objects)
            self.obstacles.append(obstacle)
            existing_objects.append(obstacle)
        self.reward = RenderableReward()
        self.update_grid()
        return self.agent.get_state()

    def render(self, plotter, log_messages):
        self.screen.fill(colors.WHITE)
        self.grid.clear()
        self.grid.add(self.agent)
        for food in self.foods:
            self.grid.add(food)
        for obstacle in self.obstacles:
            self.grid.add(obstacle)
        for food in self.foods:
            food.draw(self.screen, colors, self.scale)
        for obstacle in self.obstacles:
            obstacle.draw(self.screen, colors, self.scale)
        self.agent.draw(self.screen, colors, self.scale)
        self.draw_dish()
        self.reward.draw(self.screen, self.scale)
        self.draw_logs(log_messages)
        plotter.draw_plots(self.screen)
        pygame.display.flip()

    def draw_dish(self):
        center = (np.array(self.dish_center) * self.scale).astype(int)
        radius = int(self.dish_radius * self.scale)
        pygame.draw.circle(self.screen, colors.BLACK, center, radius, 2)

    def draw_logs(self, log_messages):
        y = self.screen.get_height() - 150
        pygame.draw.rect(self.screen, colors.GRAY, (0, y, self.screen.get_width(), 150))
        for i, msg in enumerate(log_messages[-5:]):
            img = font.render(msg, True, colors.BLACK)
            self.screen.blit(img, (10, y + 10 + i * 25))

