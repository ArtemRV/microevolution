import pygame
import numpy as np
from common.settings import Colors
from common.game_object import Environment, Organism, Food, Obstacle, Reward

# Инициализация Pygame
pygame.init()
font = pygame.font.SysFont(None, 24)

colors = Colors()

class RenderComponent:
    """Компонент для рендеринга объектов."""
    def __init__(self, color, shape='circle'):
        self.color = color
        self.shape = shape

    def draw(self, screen, pos, radius, scale=1.0):
        pos = (pos * scale).astype(int)
        radius = int(radius * scale)
        if self.shape == 'circle':
            pygame.draw.circle(screen, self.color, pos, radius)
        else:
            raise NotImplementedError(f"Shape {self.shape} not supported yet")

class RenderableReward(Reward):
    """Рендеринг наград."""
    def draw(self, screen, scale=1.0):
        y = int(10 * scale)
        texts = [
            f"Total Reward: {self.total:.2f}",
            f"Eat Reward: {self.eat:.2f}",
            f"Obstacle Collision: {self.obstacle_collision:.2f}",
            f"Dish Collision: {self.dish_collision:.2f}",
            f"Energy: {self.energy:.2f}",
            f"Approach: {self.approach:.2f}",
            f"Survival: {self.survival:.2f}"
        ]
        for i, text in enumerate(texts):
            img = font.render(text, True, colors.BLACK)
            scaled_img = pygame.transform.scale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
            screen.blit(scaled_img, (int(10 * scale), y + int(i * 20 * scale)))

class RenderableEnvironment(Environment):
    """Среда с поддержкой рендеринга."""
    def __init__(self, settings, screen, episode=0):
        # Передаем классы с поддержкой рендеринга
        super().__init__(
            settings,
            food_class=Food,
            obstacle_class=Obstacle,
            organism_class=Organism,
            episode=0,
        )
        self.screen = screen
        self.episode = episode
        self.scale = 1.0
        self.dish_center = (self.width // 2, 10 + settings['general']['dish_radius'])

        # Добавляем компоненты рендеринга
        self.agent.render_component = RenderComponent(colors.BLUE)
        for food in self.foods:
            food.render_component = RenderComponent(colors.GREEN)
        for obstacle in self.obstacles:
            obstacle.render_component = RenderComponent(colors.RED)
        self.reward = RenderableReward()

    def add_food(self, existing_objects):
        """Добавление новой еды с компонентом рендеринга."""
        new_food = Food(self, self.settings, existing_objects)
        new_food.render_component = RenderComponent(colors.GREEN)
        self.foods.append(new_food)

    def render(self, plotter, log_messages):
        """Отрисовка сцены."""
        self.screen.fill(colors.WHITE)
        self.grid.clear()
        self.grid.add(self.agent)
        for food in self.foods:
            self.grid.add(food)
        for obstacle in self.obstacles:
            self.grid.add(obstacle)
        for food in self.foods:
            if food.render_component is None:
                food.render_component = RenderComponent(colors.GREEN)
            food.render_component.draw(self.screen, food.pos, food.radius, self.scale)
        for obstacle in self.obstacles:
            if obstacle.render_component is None:
                obstacle.render_component = RenderComponent(colors.RED)
            obstacle.render_component.draw(self.screen, obstacle.pos, obstacle.radius, self.scale)
        self.agent.render_component.draw(self.screen, self.agent.pos, self.agent.radius, self.scale)
        self.draw_dish()
        self.reward.draw(self.screen, self.scale)
        self.draw_logs(log_messages)
        plotter.draw_plots(self.screen)
        pygame.display.flip()

    def draw_dish(self):
        """Отрисовка чашки Петри."""
        center = (np.array(self.dish_center) * self.scale).astype(int)
        radius = int(self.dish_radius * self.scale)
        pygame.draw.circle(self.screen, colors.BLACK, center, radius, 2)

    def draw_logs(self, log_messages):
        """Отрисовка логов."""
        y = self.screen.get_height() - 150
        pygame.draw.rect(self.screen, colors.GRAY, (0, y, self.screen.get_width(), 150))
        for i, msg in enumerate(log_messages[-5:]):
            img = font.render(msg, True, colors.BLACK)
            self.screen.blit(img, (10, y + 10 + i * 25))
