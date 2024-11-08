import pygame
import sympy as sp

# Инициализация Pygame
pygame.init()

# Настройки экрана
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Math Equation Evaluator")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32)

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Поле ввода
input_box = pygame.Rect(50, 50, 500, 32)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color = color_inactive
active = False
text = ''
result = ''

# Функция для вычисления уравнения
def evaluate_equation(equation, variables):
    try:
        # Преобразуем строку в выражение SymPy
        expr = sp.sympify(equation)
        
        # Вычисляем значение выражения с заданными переменными
        result = expr.evalf(subs=variables)
        return result
    except (sp.SympifyError, TypeError):
        return "Ошибка в уравнении."

def main():
    global active, color, text, result

    running = True
    while running:
        screen.fill(WHITE)

        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Проверка клика по полю ввода
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            elif event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        # Пример: вводим значения переменных
                        variables = {"x": 2, "y": 3}  # Замените нужными значениями
                        result = evaluate_equation(text, variables)
                        text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode

        # Рендеринг текста
        txt_surface = font.render(text, True, BLACK)
        screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(screen, color, input_box, 2)

        # Отображение результата
        result_surface = font.render(f"Результат: {result}", True, BLUE)
        screen.blit(result_surface, (50, 100))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# Запуск основной функции
main()
