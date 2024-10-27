import pygame
import numpy as np

# Инициализация Pygame
pygame.init()

# Задаем размеры окна
width, height = 800, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
# Цвета от красного до зеленого
COLORS = [(255 - i * 30, i * 30, 0) for i in range(7)]

# Функция для отрисовки нейрона
def draw_neuron(x, y, radius):
    pygame.draw.circle(screen, BLACK, (x, y), radius, 2)

# Функция для отрисовки связи между нейронами
def draw_connection(x1, y1, x2, y2, thickness):
    pygame.draw.line(screen, COLORS[thickness - 1], (x1, y1), (x2, y2), 1)
    # pygame.draw.line(screen, COLORS[thickness - 1], (x1, y1), (x2, y2), thickness)

# Загрузить матрицу из файла weights_fc1.txt
matrix = np.loadtxt("weights_fc1.txt")

# Сортируем значения матрицы от большего к меньшему
sorted_values = np.sort(matrix.flatten())[::-1]

# Определяем диапазон значений и разбиваем его на 7 равных частей
min_val, max_val = sorted_values.min(), sorted_values.max()
intervals = np.linspace(min_val, max_val, 8)  # 7 интервалов

# Функция для определения толщины линии в зависимости от веса
def get_thickness(value):
    for i in range(7):
        if intervals[i] <= value < intervals[i + 1]:
            return i + 1
    return 1  # Если значение ниже минимального интервала

# Основной цикл Pygame
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)  # Очищаем экран

    # Координаты для расположения нейронов (пример)
    matrix_width, matrix_height = matrix.shape
    neuron_positions1 = [(10 + i * 50, 10) for i in range(matrix_width)]
    neuron_positions2 = [(10 + j * 50, 500) for j in range(matrix_height)]

    # Рисуем нейроны и связи
    for i, (x1, y1) in enumerate(neuron_positions1):
        for j, (x2, y2) in enumerate(neuron_positions2):
            if i != j:
                # Получаем вес (значение матрицы) для данной связи
                weight = matrix[i % matrix_width, j % matrix_height]
                thickness = get_thickness(weight)
                draw_connection(x1, y1, x2, y2, thickness)

    for i, (x1, y1) in enumerate(neuron_positions1):
        # Рисуем нейрон
        neuron_weight = sum(matrix[i % matrix_width])  # Пример суммарного веса нейрона
        radius = max(10, neuron_weight // 100)  # Пример радиуса нейрона
        draw_neuron(x1, y1, radius)
    
    for j, (x2, y2) in enumerate(neuron_positions2):
        # Рисуем нейрон
        neuron_weight = sum(matrix[:, j % matrix_height])  # Пример суммарного веса нейрона
        radius = max(10, neuron_weight // 100)
        draw_neuron(x2, y2, radius)

    pygame.display.flip()  # Обновляем экран

# Закрываем Pygame
pygame.quit()
