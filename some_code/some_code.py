import pygame
import pygame_textinput

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Text Input Example")
clock = pygame.time.Clock()

# Инициализация текстового ввода
textinput = pygame_textinput.TextInputVisualizer()

running = True
while running:
    # Обработка событий
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

    # Обновление текста
    textinput.update(events)

    # Очистка экрана
    screen.fill((255, 255, 255))  # Тёмный фон

    # Отображение введённого текста
    screen.blit(textinput.surface, (50, 50))  # Вывод текста на экран

    # Обновление экрана
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
