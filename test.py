import pygame
import sys

# Инициализация Pygame
pygame.init()

# Размер окна
screen = pygame.display.set_mode((800, 600))

class Image(pygame.sprite.Sprite):
    def __init__(self, image, x, y, width=None, height=None):
        self.image = pygame.image.load(image).convert()
        # Set the color that will be transparent
        self.image.set_colorkey((255, 255, 255))
        # Convert the image to the same format as the screen
        self.image = self.image.convert_alpha()
        # Resize the image
        if width and height:
            self.image = pygame.transform.scale(self.image, (width, height))
        # Set sprite's position
        self.sprite_rect = self.image.get_rect(center=(x, y))
        # Create a sprite
        self.sprite_mask = pygame.mask.from_surface(self.image)

    def move(self, x, y):
        self.sprite_rect.x = x
        self.sprite_rect.y = y

    def rotate(self, angle):
        self.image = pygame.transform.rotate(self.image, angle)
        self.sprite_mask = pygame.mask.from_surface(self.image)

image = Image("organism.png", 400, 200)

# Загрузка другого объекта (например, кружок)
circle_image = pygame.Surface((50, 50), pygame.SRCALPHA)
pygame.draw.circle(circle_image, (255, 0, 0), (25, 25), 25)
circle_rect = circle_image.get_rect(center=(200, 150))

# Создание маски для кружка
circle_mask = pygame.mask.from_surface(circle_image)

# Основной игровой цикл
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Движение кружка по оси X (например, для проверки пересечения)
    circle_rect.x += 1

    # Проверка пересечения масок
    offset = (circle_rect.x - image.sprite_rect.x, circle_rect.y - image.sprite_rect.y)
    collision = image.sprite_mask.overlap(circle_mask, offset)

    # Заливка экрана белым цветом
    screen.fill((0, 0, 0))

    # Отрисовка спрайта и кружка
    screen.blit(image.image, image.sprite_rect)
    screen.blit(circle_image, circle_rect)

    # Если есть пересечение, меняем цвет экрана
    if collision:
        print("Пересечение!")
        screen.fill((255, 255, 0))  # Желтый экран при пересечении
        # Возвращаем кружок на начальную позицию
        circle_rect.x = 200

    # Обновление экрана
    pygame.display.flip()
