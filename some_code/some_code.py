import pygame
import math

# Инициализация Pygame
pygame.init()

# Размер окна
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Модель движения")

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Параметры
RADIUS = 20       # Радиус кружочков
SPEED = 3         # Скорость движения
LEG_LENGTH = 30   # Длина ноги

class organism:
    def __init__(self, x, y, angle, radius, speed, leg_length):
        self.body = [x, y]
        self.angle = angle
        self.radius = radius
        self.speed = speed
        self.left_leg1 = self.get_leg(45)
        self.right_leg1 = self.get_leg(-45)
        
    def get_leg(self, additional_angle):
        additional_angle = math.radians(additional_angle)
        x1 = self.body[0] + self.radius * (math.cos(self.angle + additional_angle))
        y1 = self.body[1] + self.radius * (math.sin(self.angle + additional_angle))
        x2 = x1 + LEG_LENGTH * (math.cos(self.angle + additional_angle))
        y2 = y1 + LEG_LENGTH * (math.sin(self.angle + additional_angle))
        return [[x1, y1], [x2, y2]]
    
    def move(self):
        x_shift = self.speed * math.cos(self.angle)
        y_shift = self.speed * math.sin(self.angle)
        self.body[0] += x_shift
        self.body[1] -= y_shift
        self.left_leg1[0][0] += x_shift
        self.left_leg1[0][1] -= y_shift
        self.right_leg1[0][0] += x_shift
        self.right_leg1[0][1] -= y_shift

    def check_leg_angle(self):
        pass

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.body[0]), int(self.body[1])), self.radius)
        pygame.draw.line(screen, WHITE, self.left_leg1[0], self.left_leg1[1], 3)
        pygame.draw.line(screen, WHITE, self.right_leg1[0], self.right_leg1[1], 3)

organism1 = organism(100, HEIGHT // 2, 0, RADIUS, SPEED, LEG_LENGTH)

# Основной цикл игры
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                organism1.angle -= math.pi / 18  # Поворот влево
            elif event.key == pygame.K_RIGHT:
                organism1.angle += math.pi / 18  # Поворот вправо

    # Отрисовка
    screen.fill(BLACK)
    
    organism1.move()
    organism1.draw()    
    
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
