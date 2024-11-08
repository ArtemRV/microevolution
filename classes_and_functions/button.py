import pygame
import yaml

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color, hover_text_color, font, action=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = self.update_rect()
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.text_color = text_color
        self.hover_text_color = hover_text_color
        self.font = font
        self.current_color = self.color
        self.current_text_color = self.text_color
        self.action = action

    def update_rect(self, x_ratio=0, y_ratio=0):
        self.rect = pygame.Rect(x_ratio + self.x, y_ratio + self.y, self.width, self.height)

    def draw(self, surface):
        pygame.draw.rect(surface, self.current_color, self.rect)
        # Отображаем текст на кнопке
        text_surface = self.font.render(self.text, True, self.current_text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def check_hover(self):
        # Смена цвета при наведении
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            self.current_color = self.hover_color
            self.current_text_color = self.hover_text_color
        else:
            self.current_color = self.color
            self.current_text_color = self.text_color

    def is_clicked(self, event):
        return self.rect.collidepoint(event.pos)
    
class Checkbox:
    def __init__(self, x, y, size, color, hover_color, check_color, text, text_color, font, checked, active, action=None):
        self.x = x
        self.y = y
        self.size = size
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        self.color = color
        self.hover_color = hover_color
        self.check_color = check_color
        self.current_color = self.color
        self.text = text
        self.text_color = text_color
        self.font = font
        self.checked = checked
        self.active = active
        self.action = action

    def update_rect(self, x_ratio=0, y_ratio=0):
        self.rect = pygame.Rect(x_ratio + self.x, y_ratio + self.y, self.size, self.size)

    def draw(self, surface):
        pygame.draw.rect(surface, self.current_color, self.rect)
        # Если флажок установлен, рисуем галочку
        if self.checked:
            pygame.draw.rect(surface, self.check_color, self.rect.inflate(-10, -10))
        # Отображаем текст
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
        surface.blit(text_surface, text_rect)

    def check_hover(self):
        # Смена цвета при наведении
        if self.active:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.current_color = self.hover_color
            else:
                self.current_color = self.color
        else:
            self.current_color = (200, 200, 200)

# Загрузка настроек кнопок из YAML
def load_ui_element_settings(element_class, colors, yaml_file, actions, element_key):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        array = []
        
        # Получаем нужный массив настроек из файла
        settings_array = data.get(element_key, [])
        
        for settings in settings_array:
            if actions is not None:
                action_func = actions.get(settings['action'])
            else:
                action_func = None
            font = pygame.font.SysFont(None, settings['font'])
            
            # Общие настройки для всех UI-элементов
            element_params = {
                'x': settings['x'],
                'y': settings['y'],
                'text': settings['text'],
                'color': getattr(colors, settings['color']),
                'hover_color': getattr(colors, settings['hover_color']),
                'text_color': getattr(colors, settings['text_color']),
                'font': font,
                'action': action_func
            }
            
            # Специфические настройки для Button
            if element_class == Button:
                element_params.update({
                    'width': settings['width'],
                    'height': settings['height'],
                    'hover_text_color': getattr(colors, settings['hover_text_color'])
                })

            # Специфические настройки для Checkbox
            elif element_class == Checkbox:
                element_params.update({
                    'size': settings['size'],
                    'check_color': getattr(colors, settings['check_color']),
                    'checked': settings['checked'],
                    'active': settings['active'],
                })

            # Создаем элемент и добавляем в массив
            element = element_class(**element_params)
            array.append(element)

        return array

# Обертки для кнопок и чекбоксов
def load_button_settings(colors, yaml_file, actions, group_name):
    return load_ui_element_settings(Button, colors, yaml_file, actions, group_name)

def load_checkbox_settings(colors, yaml_file, actions, group_name):
    return load_ui_element_settings(Checkbox, colors, yaml_file, actions, group_name)