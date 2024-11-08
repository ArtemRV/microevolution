import pygame
import torch
import threading
import queue
import time
import math

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()

# Очередь для передачи статуса обучения
status_queue = queue.Queue()

# Функция для обучения нейронной сети
def train_network(num_epochs, status_queue):
    for epoch in range(num_epochs):
        # Здесь будет фактический код обучения на Torch
        time.sleep(10)  # Симуляция времени на обучение одной эпохи
        status_queue.put(f"Epoch {epoch + 1}/{num_epochs} complete")  # Обновляем статус
    status_queue.put("Training complete")

# Функция для запуска обучения в отдельном потоке
def start_training(num_epochs):
    training_thread = threading.Thread(target=train_network, args=(num_epochs, status_queue))
    training_thread.start()

# Основная программа Pygame
def main_program():
    num_epochs = 10  # Задаем количество эпох
    start_training(num_epochs)

    angle = 0  # Угол для анимации вращения

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Отрисовка фона
        screen.fill((255, 255, 255))

        # Обработка статуса из очереди
        while not status_queue.empty():
            status = status_queue.get()
            print(status)  # Выводим статус в консоль
            if status == "Training complete":
                running = False  # Останавливаем программу после завершения

        # Анимация вращающегося индикатора
        center = (200, 200)
        radius = 50
        end_pos = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        pygame.draw.circle(screen, (0, 100, 200), center, radius, 5)
        pygame.draw.line(screen, (200, 50, 50), center, end_pos, 5)

        # Обновление угла для вращения индикатора
        angle += 0.1

        # Обновление экрана
        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main_program()
