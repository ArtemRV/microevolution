import numpy as np
from common.utils import is_position_free

class GameObject:
    """Базовый класс для игровых объектов."""
    def __init__(self, env, settings, object_type, existing_objects=None):
        self.env = env
        self.settings = settings
        self.object_type = object_type  # 'organism', 'food', or 'obstacle'
        self.radius = settings[object_type]['radius']
        self.pos = self._initialize_position(existing_objects)
        self.vel = np.array([0.0, 0.0])
        self.render_component = None  # Для рендеринга

    def _initialize_position(self, existing_objects):
        if self.object_type == 'organism' and not self.settings['organism']['random_start']:
            return np.array([self.env.dish_center[0], self.env.dish_center[1]], dtype=float)
        
        max_attempts = 100
        max_radius = (self.settings['organism']['start_radius'] if self.object_type == 'organism' 
                     else self.env.dish_radius)
        for _ in range(max_attempts):
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, max_radius - self.radius)
            pos = np.array([
                self.env.dish_center[0] + r * np.cos(theta),
                self.env.dish_center[1] + r * np.sin(theta)
            ])
            if is_position_free(pos, self.radius, self.env.dish_center, self.env.dish_radius, existing_objects or [], 5.0):
                return pos
        raise ValueError(
            f"Failed to place {self.object_type} after {max_attempts} attempts."
            "Possible reasons: too many objects, dish radius too small, or object radius too large."
        )

    def move(self):
        """Update position based on velocity and handle dish boundary collisions."""
        self.pos += self.vel
        dist_to_center = np.linalg.norm(self.pos - self.env.dish_center)
        if dist_to_center > self.env.dish_radius - self.radius:
            normal = (self.env.dish_center - self.pos) / (dist_to_center + 1e-6)
            dot_product = np.dot(self.vel, normal)
            self.vel = self.vel - 2 * dot_product * normal
            direction = (self.pos - self.env.dish_center) / (dist_to_center + 1e-6)
            self.pos = self.env.dish_center + direction * (self.env.dish_radius - self.radius)

class Organism(GameObject):
    """Класс для организма."""
    def __init__(self, env, settings):
        super().__init__(env, settings, 'organism')
        self.energy = settings['organism']['initial_energy']
        self.visible_obstacle = settings['organism']['visible_obstacle']
        self.visible_food = settings['organism']['visible_food']
        self.food_eaten = 0
        self.max_speed = settings['organism']['max_speed']
        self.prev_action = np.array([0.0, 0.0])
        self.steps_without_food = 0
        self.current_energy_penalty = settings['rewards']['energy']['value']
        self.step_count = 0  # Счетчик шагов для награды за выживание

    def reset(self):
        self.pos = self._initialize_position([])
        self.vel = np.array([0.0, 0.0])
        self.energy = self.settings['organism']['initial_energy']
        self.food_eaten = 0
        self.prev_action = np.array([0.0, 0.0])
        self.steps_without_food = 0
        self.current_energy_penalty = self.settings['rewards']['energy']['value']
        self.step_count = 0  # Сбрасываем счетчик шагов при сбросе

    def move(self, action, foods, obstacles):
        acceleration = np.array(action) * self.settings['organism']['max_acceleration']
        self.vel += acceleration
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed and speed > 0:
            self.vel = self.vel / speed * self.max_speed
        prev_pos = self.pos.copy()
        self.pos += self.vel
        self.prev_action = action
        self.step_count += 1  # Увеличиваем счетчик шагов

        if not self.settings['rewards_enabled']:
            return False

        reward_settings = self.settings.get('rewards', {})
        self.steps_without_food += 1

        """Approach Reward"""
        if reward_settings['approach']['enabled']:
            nearby_foods = [obj for obj in self.env.grid.get_nearby_objects(self.pos) if obj.object_type == 'food']
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
        nearby_foods = [obj for obj in nearby_objects if obj.object_type == 'food']
        nearby_obstacles = [obj for obj in nearby_objects if obj.object_type == 'obstacle']

        """Food reward and penalty reset"""
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
                    self.env.add_food([self] + self.env.obstacles + self.env.foods)

        """Reset energy penalty when eating"""
        if ate_food:
            self.steps_without_food = 0
            self.current_energy_penalty = reward_settings['energy']['value']

        """Growing energy penalty"""
        if reward_settings['energy']['enabled']:
            increment_steps = reward_settings['energy']['increment_steps']
            increment = reward_settings['energy']['increment']
            if self.steps_without_food >= increment_steps and self.steps_without_food % increment_steps == 0:
                self.current_energy_penalty -= increment
            self.energy -= self.settings['organism']['energy_per_step']
            self.env.reward.update(self.current_energy_penalty, 'energy')
            if self.energy <= 0:
                return True

        """Collision with an obstacle"""
        if reward_settings['obstacle_collision']['enabled']:
            for obstacle in nearby_obstacles:
                if np.linalg.norm(self.pos - obstacle.pos) < self.radius + obstacle.radius:
                    self.env.reward.update(reward_settings['obstacle_collision']['value'], 'obstacle_collision')
                    if reward_settings['obstacle_collision']['end_episode']:
                        return True

        """Reward for survival"""
        if reward_settings['survival']['enabled'] and self.energy > 0:
            self.env.reward.update(reward_settings['survival']['value'] + reward_settings['survival']['increment'] * (self.step_count // reward_settings['survival']['increment_steps']), 'survival')

        return False

    def get_state(self):
        state = []
        rel_pos = (self.pos - self.env.dish_center) / self.env.dish_radius
        state.extend(rel_pos)
        state.extend(self.vel / self.max_speed)
        state.append(self.energy / self.settings['organism']['max_energy'])
        state.append(np.linalg.norm(self.pos - self.env.dish_center) / self.env.dish_radius)
        direction = np.arctan2(self.vel[1], self.vel[0]) / np.pi if np.linalg.norm(self.vel) > 0 else 0
        state.append(direction)
        state.extend(self.prev_action)

        foods = sorted(self.env.foods, key=lambda f: np.linalg.norm(self.pos - f.pos))
        for i in range(self.visible_food):
            if i < len(foods):
                rel_food_pos = (foods[i].pos - self.pos) / self.env.dish_radius
                state.extend(rel_food_pos)
                state.extend(foods[i].vel / self.max_speed)
            else:
                state.extend([0, 0, 0, 0])

        obstacles = sorted(self.env.obstacles, key=lambda o: np.linalg.norm(self.pos - o.pos))
        for i in range(self.visible_obstacle):
            if i < len(obstacles):
                rel_obstacle_pos = (obstacles[i].pos - self.env.dish_center) / self.env.dish_radius
                state.extend(rel_obstacle_pos)
                state.extend(obstacles[i].vel / self.max_speed)
            else:
                state.extend([0, 0, 0, 0])

        return np.array(state, dtype=np.float32)

class Food(GameObject):
    """Food class."""
    def __init__(self, env, settings, existing_objects):
        super().__init__(env, settings, 'food', existing_objects)

    def move(self):
        super().move()

class Obstacle(GameObject):
    """Obstacle class."""
    def __init__(self, env, settings, existing_objects):
        super().__init__(env, settings, 'obstacle', existing_objects)
        self.vel = np.random.uniform(-settings['obstacle']['max_speed'], settings['obstacle']['max_speed'], 2)

    def move(self):
        super().move()

class Environment:
    """Базовая игровая среда."""
    def __init__(self, settings, food_class=Food, obstacle_class=Obstacle, organism_class=Organism, episode=0):
        self.settings = settings
        self.episode = episode
        self.width = settings['general']['width']
        self.height = settings['general']['height']
        self.dish_radius = settings['general']['dish_radius']
        self.dish_center = (self.width // 2, self.height // 2)
        self.food_class = food_class
        self.obstacle_class = obstacle_class
        self.organism_class = organism_class
        
        start_radius = settings['organism']['start_radius']
        if start_radius > self.dish_radius:
            raise ValueError(
                f"Organism start_radius ({start_radius}) cannot exceed dish_radius ({self.dish_radius})."
            )
        self.settings['organism']['start_radius'] = start_radius

        self.agent = self.organism_class(self, settings)
        self.grid_size = settings['general']['grid_size']
        self.grid = Grid(self.grid_size, self.width, self.height)
        self.reward = Reward()
        
        self.foods = []
        self.obstacles = []
        self._initialize_objects([self.agent])
        self.update_grid()

    def _initialize_objects(self, existing_objects):
        """Инициализация еды и препятствий."""
        object_types = [
            ('food', self.food_class, self.foods),
            ('obstacle', self.obstacle_class, self.obstacles)
        ]
        
        for entity, cls, storage in object_types:
            if self.settings[entity]['enabled']:
                storage.clear()  # Очищаем список перед добавлением новых объектов
                quantity = self._get_quantity(entity)
                for _ in range(quantity):
                    obj = cls(self, self.settings, existing_objects)
                    storage.append(obj)
                    existing_objects.append(obj)

    def add_food(self, existing_objects):
        """Добавление новой еды через указанный класс."""
        new_food = self.food_class(self, self.settings, existing_objects)
        self.foods.append(new_food)

    def update_grid(self):
        self.grid.clear()
        self.grid.add(self.agent)
        for food in self.foods:
            self.grid.add(food)
        for obstacle in self.obstacles:
            self.grid.add(obstacle)

    def reset(self):
        self.agent.reset()
        existing_objects = [self.agent]
        self._initialize_objects(existing_objects)
        self.reward.reset()
        self.update_grid()
        return self.agent.get_state()

    def _get_quantity(self, entity):
        settings = self.settings[entity]
        if self.settings['rewards_enabled'] and self.settings.get('increment_' + entity):
            return settings['start_quantity'] + (self.episode // self.settings['increment_episodes']) * settings['increment_quantity']
        return settings['quantity']

    def step(self, action):
        self.reward.reset()
        done = self.agent.move(action, self.foods, self.obstacles)
        
        for obstacle in self.obstacles:
            obstacle.move()
        
        for food in self.foods:
            for obstacle in self.obstacles:
                dist = np.linalg.norm(food.pos - obstacle.pos)
                if dist < food.radius + obstacle.radius:
                    direction = (food.pos - obstacle.pos) / (dist + 1e-6)
                    food.pos += direction * (food.radius + obstacle.radius - dist)
        
        for i, food1 in enumerate(self.foods):
            for food2 in self.foods[i+1:]:
                dist = np.linalg.norm(food1.pos - food2.pos)
                if dist < food1.radius + food2.radius and dist > 0:
                    direction = (food1.pos - food2.pos) / (dist + 1e-6)
                    overlap = food1.radius + food2.radius - dist
                    food1.pos += direction * (overlap / 2)
                    food2.pos -= direction * (overlap / 2)
        
        for food in self.foods:
            food.move()
        
        self.update_grid()
        next_state = self.agent.get_state()
        reward = self.reward.get()
        return next_state, reward, done

    def get_render_data(self):
        return {
            'agent': {'pos': self.agent.pos.tolist(), 'radius': self.agent.radius},
            'foods': [{'pos': food.pos.tolist(), 'radius': food.radius} for food in self.foods],
            'obstacles': [{'pos': obstacle.pos.tolist(), 'radius': obstacle.radius} for obstacle in self.obstacles],
            'dish': {'center': list(self.dish_center), 'radius': self.dish_radius},
            'energy': self.agent.energy,
            'food_eaten': self.agent.food_eaten,
            'reward': {
                'total': self.reward.total,
                'eat': self.reward.eat,
                'obstacle_collision': self.reward.obstacle_collision,
                'dish_collision': self.reward.dish_collision,
                'energy': self.reward.energy,
                'approach': self.reward.approach,
                'survival': self.reward.survival
            }
        }

class Grid:
    def __init__(self, cell_size, width, height):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.grid = {}

    def clear(self):
        self.grid.clear()

    def add(self, obj):
        cell_x = int(obj.pos[0] // self.cell_size)
        cell_y = int(obj.pos[1] // self.cell_size)
        key = (cell_x, cell_y)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(obj)

    def get_nearby_objects(self, pos):
        cell_x = int(pos[0] // self.cell_size)
        cell_y = int(pos[1] // self.cell_size)
        objects = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cell_x + dx, cell_y + dy)
                if key in self.grid:
                    objects.extend(self.grid[key])
        return objects

class Reward:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.eat = 0
        self.obstacle_collision = 0
        self.dish_collision = 0
        self.energy = 0
        self.approach = 0
        self.survival = 0

    def update(self, value, reward_type):
        self.total += value
        if reward_type == 'eat':
            self.eat += value
        elif reward_type == 'obstacle_collision':
            self.obstacle_collision += value
        elif reward_type == 'dish_collision':
            self.dish_collision += value
        elif reward_type == 'energy':
            self.energy += value
        elif reward_type == 'approach':
            self.approach += value
        elif reward_type == 'survival':
            self.survival += value

    def get(self):
        return self.total