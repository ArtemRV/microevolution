# Текстовый вывод
def render_text(screen, text, x, y):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        img = font.render(line, True, BLACK)
        screen.blit(img, (x, y + i * 20))

def get_input_text(input_vector):
    text = "Input vector:\n"
    text += f"food_x: {input_vector[0]:.2f} food_y: {input_vector[1]:.2f}\n"
    text += f"obstacle1_x: {input_vector[2]:.2f} obstacle1_y: {input_vector[3]:.2f}\n"
    text += f"obstacle2_x: {input_vector[4]:.2f} obstacle2_y: {input_vector[5]:.2f}\n"
    text += f"obstacle3_x: {input_vector[6]:.2f} obstacle3_y: {input_vector[7]:.2f}\n"
    text += f"vx: {input_vector[8]:.2f} vy: {input_vector[9]:.2f}\n"
    text += f"energy: {input_vector[10]:.2f}\n"
    text += f"distance_to_boundary: {input_vector[11]:.2f}\n"
    return text

# Render input vector
if organism.input_vector is not None:
    input_text = get_input_text(organism.input_vector)
    render_text(screen, input_text, WIDTH - 300, 20)




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


# Оптимизация логики столкновений с использованием сетки
class Grid:
    def __init__(self, width, height, cell_size):
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.grid = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

    def add_object(self, obj):
        col, row = int(obj.x // self.cell_size), int(obj.y // self.cell_size)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.grid[row][col].append(obj)

    def remove_object(self, obj):
        col, row = int(obj.x // self.cell_size), int(obj.y // self.cell_size)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.grid[row][col].remove(obj)

    def get_nearby_objects(self, x, y, radius):
        col, row = int(x // self.cell_size), int(y // self.cell_size)
        nearby_objects = []
        for r in range(max(0, row - 1), min(self.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.cols, col + 2)):
                nearby_objects.extend(self.grid[r][c])
        return [obj for obj in nearby_objects if np.hypot(obj.x - x, obj.y - y) < radius]
    

obstacles = [Obstacle() for _ in range(OBSTACLE_QUANTITY)]
foods = [Food() for _ in range(FOOD_QUANTITY)]
grid = Grid(WIDTH, HEIGHT, 100)  # Создание сетки для оптимизации столкновений




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Определение архитектуры нейронной сети для DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(12, 64)  # Входной слой (12 параметров)
        self.fc2 = nn.Linear(64, 32)  # Скрытый слой
        self.fc3 = nn.Linear(32, 16)  # Скрытый слой
        self.fc4 = nn.Linear(16, 2)   # Выходной слой (две координаты для перемещения)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Инициализация DQN модели, оптимизатора и функции потерь
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Параметры обучения
gamma = 0.99  # Фактор дисконтирования
epsilon = 0.1  # Вероятность случайного действия (эксплорейшн)
memory = []  # Опыт для реплея
batch_size = 64  # Размер батча для обучения
max_memory = 1000  # Максимальный размер памяти

# Функция получения награды
def get_reward(state, action, next_state, done):
    reward = 0
    if done:
        reward -= 100  # Штраф за столкновение или потерю энергии
    else:
        distance_to_target = np.linalg.norm(np.array(next_state[8:10]) - np.array(next_state[6:8]))  # Расстояние до цели
        reward += 10 / (distance_to_target + 1)  # Чем ближе к цели, тем больше награда
        if next_state[-1] < state[-1]:  # Проверка уровня энергии
            reward -= 1  # Штраф за уменьшение энергии
    return reward

# Процесс обучения
for episode in range(1000):  # Количество эпизодов
    state = env.reset()  # Начальное состояние (например, координаты, энергия и т.д.)
    done = False
    total_reward = 0
    
    while not done:
        # Эпсилон-гриди выбор действия
        if random.uniform(0, 1) < epsilon:
            action = np.random.uniform(-1, 1, size=2)  # Случайное действие
        else:
            action = model(torch.tensor(state, dtype=torch.float32)).detach().numpy()  # Прогноз нейронной сети
        
        next_state, done = env.step(action)  # Получаем новое состояние от среды
        reward = get_reward(state, action, next_state, done)  # Рассчитываем награду
        
        # Сохраняем опыт
        memory.append((state, action, reward, next_state, done))
        if len(memory) > max_memory:
            memory.pop(0)
        
        state = next_state
        total_reward += reward

        # Обучение модели
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            for s, a, r, ns, d in batch:
                target = r + gamma * model(torch.tensor(ns, dtype=torch.float32)).max().item() * (1 - d)
                pred = model(torch.tensor(s, dtype=torch.float32))
                loss = loss_fn(pred, torch.tensor(a, dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print(f'Episode {episode}, Total Reward: {total_reward}')
