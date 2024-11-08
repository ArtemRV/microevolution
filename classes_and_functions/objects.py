import numpy as np
import pygame
import torch


class Grid:
    def __init__(self, grid_size):
        self.grid = {}
        self.size = grid_size

    def add_to_grid(self, obj):
        grid_x, grid_y = self.get_grid_x_y(obj)
        self.grid.setdefault((grid_x, grid_y), []).append(obj)

    def get_nearby_objects(self, obj):
        grid_x, grid_y = self.get_grid_x_y(obj)
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nearby.extend(self.grid.get((grid_x + dx, grid_y + dy), []))
        return nearby
    
    def get_grid_x_y(self, obj):
        grid_x = int(obj.x // self.size)
        grid_y = int(obj.y // self.size)
        return grid_x, grid_y

class Dish:
    def __init__(self, radius, dish_collision_penalty, penalty_max, screen):
        self.get_center(screen)
        self.radius = radius
        self.dish_collision_penalty = dish_collision_penalty
        self.penalty_max = penalty_max

    def get_center(self, screen):
        self.x, self.y = screen.get_width() // 2, screen.get_height() // 2

    def draw(self, screen, colors):
        pygame.draw.circle(screen, colors.BLUE, (int(self.x), int(self.y)), self.radius, 2)

class Obstacle:
    def __init__(self, obstacle_radius, dish, organism, obstacles, obstacle_collision_penalty, obstacle_speed):
        self.radius = obstacle_radius
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.speed = obstacle_speed
        while True:
            x, y = random_x_y(dish, self.radius)

            if not check_collision(x, y, self.radius, organism.x, organism.y, organism.radius):
                if not any(check_collision(x, y, self.radius, o.x, o.y, self.radius) for o in obstacles):
                    self.x, self.y = x, y
                    self.vx, self.vy = np.random.uniform(-self.speed, self.speed, 2)
                    break

    def move(self, dish, grid):
        # Проверка столкновений с чашкой Петри и другими препятствиями
        if check_dish_collision(self.x + self.vx, self.y + self.vy, self.radius, dish):
            angle_rebound = calculate_rebound_angle(self.x, self.y, 0, 0, self.vx, self.vy)
            self.vx = -np.cos(angle_rebound) * np.hypot(self.vx, self.vy)
            self.vy = -np.sin(angle_rebound) * np.hypot(self.vx, self.vy)

        self.x += self.vx
        self.y += self.vy

        nearby = grid.get_nearby_objects(self)
        check_repel(self, dish, nearby)

    def draw(self, screen, colors, dish):
        pygame.draw.circle(screen, colors.RED, (int(self.x + dish.x), int(self.y + dish.y)), self.radius)

class Food:
    def __init__(self, food_radius, food_energy, dish, organism, obstacles, foods, food_reward):
        self.radius = food_radius
        self.energy = food_energy
        self.food_reward = food_reward
        while True:
            x, y = random_x_y(dish, self.radius)

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

    def draw(self, screen, colors, dish):
        pygame.draw.circle(screen, colors.GREEN, (int(self.x + dish.x), int(self.y + dish.y)), self.radius)

    def check_eaten(self, x, y, r2):
        # Проверка столкновения между организмом и едой
        distance = np.hypot(self.x - x, self.y - y)
        return distance < (self.radius + r2)
    
# Класс для организма
class Organism:
    def __init__(self, dish, organism_radius, speed, color, initial_energy, energy_loss_rate, speed_energy_loss_multiplier, energy_depletion_penalty):
        self.radius = organism_radius
        self.x, self.y = random_x_y(dish, self.radius)
        self.old_x = self.x
        self.old_y = self.y
        self.speed = speed
        self.color = color
        self.vx = np.random.uniform(-self.speed, self.speed)
        self.vy = np.random.uniform(-self.speed, self.speed)
        self.energy_loss_rate = energy_loss_rate
        self.speed_energy_loss_multiplier = speed_energy_loss_multiplier
        self.energy_depletion_penalty = energy_depletion_penalty
        self.initial_energy = initial_energy
        self.energy = initial_energy
        self.input_vector = None
        self.reset_status = False
        self.done = False
        self.reset_counters()

    def reset_counters(self):
        """Reset the organism's counters."""
        self.food_eaten = 0
        self.death_counter = 0
        self.dish_collision_counter = 0
        self.obstacle_collision_counter = 0
        self.energy_loss_counter = 0

    # Generate the input vector for the neural network
    def get_input_vector(self, dish, obstacles=None, foods=None):
        self.input_vector = []
        if foods is not None:
            self.closest_food = sorted(foods, key=lambda food: np.hypot(self.x - food.x, self.y - food.y))[:3]
            for food in self.closest_food:
                self.input_vector.append(food.x - self.x)
                self.input_vector.append(food.y - self.y)

        self.input_vector.append(self.vx)
        self.input_vector.append(self.vy)
        self.input_vector.append(np.hypot(self.x, self.y) + self.radius - dish.radius)
        self.input_vector.append(self.energy)

        if obstacles is not None:
            closest_obstacles = sorted(obstacles, key=lambda obstacle: np.hypot(self.x - obstacle.x, self.y - obstacle.y))[:3]
            self.closest_obstacles = closest_obstacles
            for obstacle in closest_obstacles:
                self.input_vector.append(self.x - obstacle.x)
                self.input_vector.append(self.y - obstacle.y)
                self.input_vector.append(obstacle.vx)
                self.input_vector.append(obstacle.vy)
        
        # self.input_vector = np.array([
        #     self.closest_food[0].x - self.x, self.closest_food[0].y - self.y,
        #     self.closest_food[1].x - self.x, self.closest_food[1].y - self.y,
        #     self.closest_food[2].x - self.x, self.closest_food[2].y - self.y,
            # self.vx, self.vy,
            # np.hypot(self.x, self.y) + self.radius - dish.radius,
            # self.energy,
        #     self.x - closest_obstacles[0].x, self.y - closest_obstacles[0].y,
        #     closest_obstacles[0].vx, closest_obstacles[0].vy,
        #     self.x - closest_obstacles[1].x, self.y - closest_obstacles[1].y,
        #     closest_obstacles[1].vx, closest_obstacles[1].vy,
        #     self.x - closest_obstacles[2].x, self.y - closest_obstacles[2].y,
        #     closest_obstacles[2].vx, closest_obstacles[2].vy
        # ])
        return torch.tensor(self.input_vector, dtype=torch.float32)

    def move(self, action, dish, obstacles, reward, foods):
        """Move the organism based on the action."""
        self.done = False
        self.reset_status = False
        action_tensor = action.clone().detach().view(-1)
        new_vx = action_tensor[0].item() * self.speed
        new_vy = action_tensor[1].item() * self.speed

        self.old_x, self.old_y = self.x, self.y
        self.x += new_vx
        self.y += new_vy

        self.vx, self.vy = new_vx, new_vy

        self.dish_collision(reward, dish)
        self.obstacle_collision(reward)
        self.food_collision(reward, dish, obstacles, foods)
        self.energy_update(reward)

        if self.reset_status:
            self.reset(dish, obstacles)

        return self.done

    def food_collision(self, reward, dish, obstacles, foods):
        """Handle food collision and update rewards."""
        for i, food in enumerate(self.closest_food):
            if food.check_eaten(self.x, self.y, self.radius):
                self.energy += food.energy
                foods.remove(food)
                foods.append(Food(food.radius, food.energy, dish, self, obstacles, foods, food.food_reward))
                self.food_eaten += 1
            new_reward = self.food_reward(food, i, self.x, self.y, self.vx, self.vy)
            reward.update(new_reward, 'eat')

    def food_reward(self, food, i, x, y, vx, vy):
        """Calculate the reward for food collision."""
        if food.check_eaten(x, y, self.radius):
            return food.food_reward
        else:
            return (3 - i) / 3 * calculate_reward(self.old_x, self.old_y, vx, vy, food.x, food.y, self.radius, food.radius, 75)

    def obstacle_collision(self, reward):
        """Handle obstacle collision and update rewards."""
        for i, obstacle in enumerate(self.closest_obstacles):
            if check_collision(self.x, self.y, self.radius, obstacle.x, obstacle.y, obstacle.radius):
                self.obstacle_collision_counter += 1
                self.reset_status = True
            new_reward = self.obstacle_reward(obstacle, i, self.x, self.y, self.vx, self.vy)
            reward.update(new_reward, 'obstacle_collision')

    def obstacle_reward(self, obstacle, i, x, y, vx, vy):
        """Calculate the reward for obstacle collision."""
        if check_collision(x, y, self.radius, obstacle.x, obstacle.y, obstacle.radius):
            return obstacle.obstacle_collision_penalty
        else:
            return -(3 - i) / 3 * calculate_reward(self.old_x, self.old_y, vx, vy, obstacle.x, obstacle.y, self.radius, obstacle.radius, 100)

    def energy_update(self, reward):
        """Update the organism's energy and handle depletion."""
        self.energy -= self.energy_loss_rate + np.hypot(self.vx, self.vy) * self.speed_energy_loss_multiplier
        if self.energy <= 0:
            new_reward = self.energy_depletion_penalty
            self.energy_loss_counter += 1
            self.reset_status = True
        else:
            new_reward = calculate_simple_reward(self.energy, -7)
        reward.update(new_reward, 'energy')
    
    def dish_collision(self, reward, dish):
        """Handle dish collision and update rewards."""
        overlap = self.dish_overlap(self.vx, self.vy, dish)
        new_reward = self.dish_reward(overlap, dish)
        if overlap > 0:
            self.dish_collision_counter += 1
            self.reset_status = True
        reward.update(new_reward, 'dish_collision')

    def dish_reward(self, overlap, dish):
        """Calculate the reward for dish collision."""
        previous_distance = calculate_distance(self.old_x, self.old_y, 0, 0) - dish.radius + self.radius
        
        if overlap > 0:
            new_reward = dish.dish_collision_penalty
        elif overlap > dish.penalty_max and previous_distance - overlap > 0:
            new_reward = abs(previous_distance - overlap) * 2
        elif overlap > dish.penalty_max and previous_distance - overlap < 0:
            new_reward = -abs(previous_distance - overlap) * 15
        else:
            new_reward = 0
        return new_reward
    
    def dish_overlap(self, vx, vy, dish):
        """Calculate the overlap with the dish boundary."""
        overlap = calculate_distance(self.old_x + vx, self.old_y + vy, 0, 0) - dish.radius + self.radius
        return overlap

    def reset(self, dish, obstacles):
        """Reset the organism's position and state."""
        while True:
            self.x, self.y = random_x_y(dish, self.radius)
            collision = False
            for obstacle in obstacles:
                if check_collision(self.x, self.y, self.radius, obstacle.x, obstacle.y, obstacle.radius):
                    collision = True
                    break
            if not collision:
                break
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-2, 2)
        self.energy = self.initial_energy
        self.done = True
        self.death_counter += 1

    def draw(self, screen, colors, dish):
        pygame.draw.circle(screen, self.color, (int(self.x + dish.x), int(self.y + dish.y)), self.radius)


def random_x_y(dish, radius):
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, dish.radius - radius)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y

def calculate_distance(x1, y1, x2, y2):
    return np.hypot(x1 - x2, y1 - y2)

def check_dish_collision(x, y, r, dish):
    return calculate_distance(x, y, 0, 0) + r > dish.radius

def check_collision(x1, y1, r1, x2, y2, r2):
    return calculate_distance(x1, y1, x2, y2) < r1 + r2

def calculate_rebound_angle(x1, y1, x2, y2, vx, vy):
    angle = np.arctan2(y1 - y2, x1 - x2)
    vel_angle = np.arctan2(vy, vx)
    rebound_angle = 2 * angle - vel_angle
    return rebound_angle

# Функция для отталкивания объектов
def repel(dish, moving_obj, static_obj, nearby):
    dx, dy = static_obj.x - moving_obj.x, static_obj.y - moving_obj.y
    distance = np.hypot(dx, dy)
    overlap = moving_obj.radius + static_obj.radius - distance

    # Отталкиваем объект на величину overlap в направлении от движущегося объекта
    if distance != 0:
        direction = np.array([dx / distance, dy / distance])  # Нормализованный вектор
        if check_dish_collision(static_obj.x + direction[0] * overlap, static_obj.y + direction[1] * overlap, static_obj.radius, dish):
            # Угол к центру чашки Петри
            angle_to_center = np.arctan2(static_obj.y, static_obj.x)

            # Касательное направление (перпендикулярное к радиальному)
            tangent_dx = -np.sin(angle_to_center)
            tangent_dy = np.cos(angle_to_center)

            # Определяем направление сдвига вдоль касательной
            # Если скалярное произведение положительно — оставляем направление, если отрицательно — инвертируем
            if np.dot(direction, [tangent_dx, tangent_dy]) < 0:
                tangent_dx, tangent_dy = -tangent_dx, -tangent_dy

            # Максимальное допустимое расстояние с учетом радиуса
            max_distance = dish.radius - static_obj.radius
            static_obj.x = max_distance * np.cos(angle_to_center) + tangent_dx * overlap
            static_obj.y = max_distance * np.sin(angle_to_center) + tangent_dy * overlap
        else:    
            static_obj.x += direction[0] * overlap
            static_obj.y += direction[1] * overlap
        check_repel(static_obj, dish, nearby)

def check_repel(obj, dish, nearby):
    for other in nearby[:]:
        if other == obj:
            continue
        if check_collision(obj.x, obj.y, obj.radius, other.x, other.y, other.radius):
            if obj in nearby:
                nearby.remove(obj)
            repel(dish, obj, other, nearby)

# Reward functions
def calculate_simple_reward(parameter, factor):
    calculated_reward = -7 / abs(parameter)
    if calculated_reward < factor:
        calculated_reward = factor
    return calculated_reward

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