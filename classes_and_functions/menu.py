import pygame
import sys
from functools import partial
from classes_and_functions.button import load_button_settings, load_checkbox_settings

# Функция для игры
def game(settings, checkboxes):
    for checkbox in checkboxes:
        settings[checkbox.text] = checkbox.checked

# Функция для выхода
def quit_game():
    pygame.quit()
    sys.exit()

def food(checkboxes):
    # Проверка, что checkboxes не пустой
    if checkboxes:
        for checkbox in checkboxes:
            if checkbox.text == "3 nearest food":
                checkbox.active = not checkbox.active

def menu(screen, colors, clock, settings=None):
    # Загружаем чекбоксы
    checkboxes = load_checkbox_settings(colors, "settings/checkboxes.yml", None, 'checkboxes')

    if settings is None:
        settings = {}
    else:
        for checkbox in checkboxes:
            checkbox.checked = settings.get(checkbox.text, False)

    for checkbox in checkboxes:
        if checkbox.text == "Food":
            checkbox.action = partial(food, checkboxes)

    # Настройка кнопок
    BUTTON_ACTIONS = {
        'game': partial(game, settings, checkboxes),
        'quit': quit_game
    }
    buttons = load_button_settings(colors, "settings/buttons.yml", BUTTON_ACTIONS, "menu_buttons")

    while True:
        screen.fill(colors.WHITE)

        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                done = False
                for button in buttons:
                    if button.rect.collidepoint(event.pos):
                        button.action()
                        if button.text == "Game":
                            return settings
                        done = True
                        break
                if not done:
                    for checkbox in checkboxes:
                        if checkbox.rect.collidepoint(event.pos) and checkbox.active:
                            checkbox.checked = not checkbox.checked
                            if checkbox.action:
                                checkbox.action()
                            break
        
        # Отрисовка кнопок
        for button in buttons:
            button.update_rect(screen.get_width())
            button.check_hover()
            button.draw(screen)

        # Отрисовка чекбоксов
        for checkbox in checkboxes:
            checkbox.update_rect(screen.get_width())
            checkbox.check_hover()
            checkbox.draw(screen)

        pygame.display.flip()
        clock.tick(60)
