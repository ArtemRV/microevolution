import pygame
import numpy as np
from common.settings import Colors
from common.core import Environment, Organism, Food, Obstacle, Reward, Grid

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
        acceleration = np.array(action) * self.settings['max_acceleration']
        self.vel += acceleration
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed and speed > 0:
            self.vel = self.vel / speed * self.max_speed
        self.pos += self.vel
        self.prev_action = action

        dist_to_center = np.linalg.norm(self.pos - self.env.dish_center)
        if dist_to_center > self.env.dish_radius - self.radius:
            self.env.reward.update(-20, 'dish_collision')
            return True

        nearby_objects = self.env.grid.get_nearby_objects(self.pos)
        nearby_foods = [obj for obj in nearby_objects if isinstance(obj, (Food, RenderableFood))]
        nearby_obstacles = [obj for obj in nearby_objects if isinstance(obj, (Obstacle, RenderableObstacle))]

        for food in nearby_foods[:]:
            if np.linalg.norm(self.pos - food.pos) < self.radius + food.radius:
                self.env.reward.update(10, 'eat')
                self.energy += self.settings['energy_per_food']
                self.food_eaten += 1
                if food in self.env.foods:
                    self.env.foods.remove(food)
                new_food = RenderableFood(self.env, self.settings, [self] + self.env.obstacles + self.env.foods)
                self.env.foods.append(new_food)

        for obstacle in nearby_obstacles:
            if np.linalg.norm(self.pos - obstacle.pos) < self.radius + obstacle.radius:
                self.env.reward.update(-20, 'obstacle_collision')
                return True

        self.energy -= self.settings['energy_per_step']
        self.env.reward.update(-self.settings['energy_per_step'], 'energy')
        if self.energy <= 0:
            return True

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
        self.width = settings['width']
        self.height = settings['height']
        self.dish_radius = settings['dish_radius']
        self.dish_center = (self.width // 2, 10 + settings['dish_radius'])
        self.grid_size = settings['grid_size']
        self.grid = Grid(self.grid_size, self.width, self.height)
        self.scale = 1.0
        
        # Use rendering-specific classes
        obstacle_quantity = min(settings['obstacle_quantity'], 2 + episode // 200)
        self.agent = RenderableOrganism(self, settings)
        self.foods = []
        existing_objects = [self.agent]
        for _ in range(settings['food_quantity']):
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
        for _ in range(self.settings['food_quantity']):
            food = RenderableFood(self, self.settings, existing_objects)
            self.foods.append(food)
            existing_objects.append(food)
        self.obstacles = []
        obstacle_quantity = min(self.settings['obstacle_quantity'], 2 + self.episode // 200)
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

