import pygame
import numpy as np

# Инициализация Pygame
pygame.init()

# Размеры окна
WIDTH, HEIGHT = 1400, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Kinematics Simulation")

# Цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Параметры суставов
joint_radius = 10  # Радиус суставов
link_length = 30  # Длина плеча
max_angle = np.radians(45)  # Максимальный угол сгиба в радианах

# Инициализация суставов
joints = [
    np.array([WIDTH // 2, HEIGHT // 2]),
    np.array([WIDTH // 2 - link_length, HEIGHT // 2]),
    np.array([WIDTH // 2 - 2 * link_length, HEIGHT // 2]),
    np.array([WIDTH // 2 - 3 * link_length, HEIGHT // 2]),
    np.array([WIDTH // 2 - 4 * link_length, HEIGHT // 2]),
    np.array([WIDTH // 2 - 5 * link_length, HEIGHT // 2]),
]

# Переменные управления
dragging_joint = None

class JointChain:
    def __init__(self, joints, link_length, max_angle, angle, speed):
        self.joints = [np.array(joint) for joint in joints]
        self.link_length = link_length
        self.max_angle = max_angle
        self.angle = angle
        self.speed = speed

    def constrain_distance(self, point1, point2, distance):
        direction = point2 - point1
        if np.linalg.norm(direction) == 0:
            return point2
        return point1 + direction / np.linalg.norm(direction) * distance

    def resolve_pass(self, start, end, step):
        for i in range(start, end, step):
            if 0 <= i - step < len(self.joints):
                self.joints[i] = self.constrain_distance(self.joints[i - step], self.joints[i], self.link_length)

            if 0 <= i + step < len(self.joints) and 0 <= i - step < len(self.joints):
                self.joints[i + step] = constrain_angle(
                    self.joints[i - step], self.joints[i], self.joints[i + step], self.max_angle
                )

    def resolve_kinematics(self, target_index, target_position, mouse_drag):
        self.joints[target_index] = np.array(target_position)
        if mouse_drag and 0 < target_index < len(self.joints) - 1:
            self.resolve_pass(target_index + 1, len(self.joints), 1)
            self.resolve_pass(target_index, -1, -1)
        else:
            self.resolve_pass(target_index + 1, len(self.joints), 1)
            self.resolve_pass(target_index - 1, -1, -1)

    def draw(self, screen):
        for i in range(len(self.joints) - 1):
            pygame.draw.line(screen, WHITE, self.joints[i], self.joints[i + 1], 2)
        for joint in self.joints:
            color = GREEN if joint is self.joints[0] else RED
            pygame.draw.circle(screen, color, joint.astype(int), joint_radius)

    def move_body(self):
        x_shift = self.speed * np.cos(self.angle)
        y_shift = self.speed * np.sin(self.angle)
        new_joint = self.joints[0] + np.array([x_shift, -y_shift])
        return new_joint

# def constrain_distance(point1, point2, distance):
#     """Смещает point2 так, чтобы расстояние между точками равнялось заданному расстоянию."""
#     direction = point2 - point1
#     if np.linalg.norm(direction) == 0:
#         return point2
#     return point1 + direction / np.linalg.norm(direction) * distance

def constrain_angle(base, joint, next_joint, max_angle):
    """
    Ограничивает угол между двумя плечами, чтобы он не превышал max_angle.
    """
    # Векторы из базового сустава
    vec1 = joint - base
    vec2 = next_joint - joint

    # Нормализация векторов
    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)

    # Текущий угол между векторами
    angle = angle_between_vectors(norm_vec1, norm_vec2)

    # Если угол больше допустимого, скорректируем
    if angle > max_angle:
        # Угол коррекции
        correction_angle = angle - max_angle

        # Определение направления поворота
        cross_product = np.cross(np.append(norm_vec1, 0), np.append(norm_vec2, 0))[2]
        rotation_sign = -1 if cross_product > 0 else 1

        # Матрица поворота
        rotation_matrix = np.array([
            [np.cos(rotation_sign * correction_angle), -np.sin(rotation_sign * correction_angle)],
            [np.sin(rotation_sign * correction_angle),  np.cos(rotation_sign * correction_angle)],
        ])

        # Корректировка позиции
        corrected_vec2 = np.dot(rotation_matrix, norm_vec2) * np.linalg.norm(vec2)
        corrected_position = joint + corrected_vec2
        return corrected_position

    return next_joint

def angle_between_vectors(vec1, vec2):
    """
    Вычисляет угол между двумя векторами в радианах.
    :param vec1: Вектор 1 (numpy array).
    :param vec2: Вектор 2 (numpy array).
    :return: Угол в радианах.
    """
    dot_product = np.dot(vec1, vec2)  # Скалярное произведение
    norm_a = np.linalg.norm(vec1)    # Длина вектора vec1
    norm_b = np.linalg.norm(vec2)    # Длина вектора vec2
    
    # Угол в радианах
    angle = np.arccos(np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0))
    return angle

class KinematicsUtils:
    @staticmethod
    def constrain_distance(point1, point2, distance):
        direction = point2 - point1
        if np.linalg.norm(direction) == 0:
            return point2
        return point1 + direction / np.linalg.norm(direction) * distance

    @staticmethod
    def angle_between_vectors(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return np.arccos(np.clip(dot_product / (norm_a * norm_b), -1.0, 1.0))

# def resolve_pass(start, end, step):
#     """
#     Проводит прямой или обратный проход по суставам, корректируя их расстояния и углы.
#     """
#     for i in range(start, end, step):
#         # Ограничение расстояния между i-1 и i
#         if 0 <= i - step < len(joints):
#             joints[i] = KinematicsUtils.constrain_distance(joints[i - step], joints[i], link_length)

#         # Ограничение угла между i-1, i и i+1
#         if 0 <= i + step < len(joints) and 0 <= i - step < len(joints):
#             joints[i + step] = constrain_angle(joints[i - step], joints[i], joints[i + step], max_angle)

# def resolve_kinematics(target_joint_index, target_position, mouse_drag):
#     """
#     Основная функция разрешения кинематики.
#     """
#     # Перемещаем выбранный сустав на заданную позицию
#     joints[target_joint_index] = np.array(target_position)

#     # Корректируем порядок проходов в зависимости от перетаскивания
#     if mouse_drag and target_joint_index > 0 and target_joint_index < len(joints) - 1:
#         # Сначала прямой проход, затем обратный
#         resolve_pass(target_joint_index + 1, len(joints), 1)  # Прямой проход
#         resolve_pass(target_joint_index, -1, -1)          # Обратный проход
#     else:
#         # Для движения головы и крайних элементов
#         resolve_pass(target_joint_index + 1, len(joints), 1)
#         resolve_pass(target_joint_index - 1, -1, -1)

# Основной цикл программы
running = True
clock = pygame.time.Clock()
speed = 3
angle = np.radians(0)
chain = JointChain(joints, link_length, max_angle, angle, speed)

while running:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Захват сустава мышкой
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for i, joint in enumerate(joints):
                if np.linalg.norm(joint - mouse_pos) < joint_radius:
                    dragging_joint = i
                    break

        # Отпускание сустава
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_joint = None

        # Управление первым суставом стрелками
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                chain.angle += np.pi / 10
            elif event.key == pygame.K_RIGHT:
                chain.angle -= np.pi / 10
            elif event.key == pygame.K_UP:
                if chain.speed < 4:
                    chain.speed += 1
            elif event.key == pygame.K_DOWN:
                if chain.speed > 0:
                    chain.speed -= 1

    # Перемещение выбранного сустава
    if dragging_joint is not None:
        chain.resolve_kinematics(dragging_joint, pygame.mouse.get_pos(), True)
    else:
        chain.resolve_kinematics(0, chain.move_body(), False)

    # Рисование суставов и линий
    chain.draw(screen)
    # for i in range(len(joints) - 1):
    #     pygame.draw.line(screen, WHITE, joints[i], joints[i + 1], 2)

    # for joint in joints:
    #     if joint is joints[0]:
    #         pygame.draw.circle(screen, GREEN, joint.astype(int), joint_radius)
    #     else:
    #         pygame.draw.circle(screen, RED, joint.astype(int), joint_radius)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
