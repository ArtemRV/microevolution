import pygame
import sys
from functools import partial
from classes_and_functions.button import load_button_settings, load_checkbox_settings, load_input_box_settings

# Функция для игры
def game(settings, checkboxes, input_boxes):
    for checkbox in checkboxes:
        settings[checkbox.text] = checkbox.checked
    for input_box in input_boxes:
        settings[input_box.description] = int(input_box.textinput.value)

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
    # Загружаем поля ввода
    INPUT_BOXES_ACTIONS = {}
    input_boxes = load_input_box_settings(colors, "settings/input_boxes.yml", INPUT_BOXES_ACTIONS, "input_boxes")

    # Загружаем чекбоксы
    checkboxes = load_checkbox_settings(colors, "settings/checkboxes.yml", None, 'checkboxes')

    if settings is None:
        settings = {}
    else:
        for checkbox in checkboxes:
            checkbox.checked = settings.get(checkbox.text, False)
        for input_box in input_boxes:
            input_box.textinput.value = str(settings.get(input_box.description, ''))

    for checkbox in checkboxes:
        if checkbox.text == "Food":
            checkbox.action = partial(food, checkboxes)

    # Настройка кнопок
    BUTTON_ACTIONS = {
        'game': partial(game, settings, checkboxes, input_boxes),
        'quit': quit_game
    }
    buttons = load_button_settings(colors, "settings/buttons.yml", BUTTON_ACTIONS, "menu_buttons")

    while True:
        screen.fill(colors.WHITE)

        # Обработка событий
        events = pygame.event.get()
        for event in events:
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
                for input_box in input_boxes:
                    input_box.handle_mouse_down(event)

        for input_box in input_boxes:
            input_box.handle_key_down(events)
        
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

        # Отрисовка полей ввода
        for input_box in input_boxes:
            input_box.update_rect(screen.get_width())
            # input_box.handle_event(pygame.event.get())
            input_box.draw(screen)

        pygame.display.flip()
        clock.tick(60)
