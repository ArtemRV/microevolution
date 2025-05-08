import sys
import numpy as np
import uuid
import asyncio
import platform
import pygame
import pygame_menu
import os
from pygame_menu.locals import INPUT_INT, INPUT_FLOAT, INPUT_TEXT, ALIGN_LEFT
from client.plotting import Plotter, test_policy
from client.main import RenderableEnvironment
from client.ddpg import DDPGAgent
from common.settings import default_settings, client_settings
from common.utils import logging
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard
writer = SummaryWriter(log_dir="output/runs/micro_evolution_" + str(uuid.uuid4()))

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = (default_settings['width'], default_settings['height'])
FPS = 120
THEME = pygame_menu.themes.THEME_SOLARIZED

class Scene:
    """Base class for scenes."""
    def __init__(self, screen, scene_manager, menu=None):
        self.screen = screen
        self.scene_manager = scene_manager
        self.menu = menu

    def handle_event(self, event):
        """Handle events."""
        if self.menu:
            self.menu.update([event])

    def draw(self):
        """Draw the scene."""
        if self.menu:
            self.menu.draw(self.screen)
        pygame.display.flip()

    def update(self):
        """Update the scene state."""
        pass

class MainMenuScene(Scene):
    """Main menu scene."""
    def __init__(self, screen, scene_manager):
        super().__init__(screen, scene_manager)
        self.menu = pygame_menu.Menu('Micro Evolution Menu', WINDOW_SIZE[0], WINDOW_SIZE[1], theme=THEME)
        self.menu.add.button('Load Model', self.switch_to_load_model)
        self.menu.add.button('Adjust Settings', self.switch_to_settings)
        self.menu.add.button('Start Simulation', self.start_simulation)
        self.menu.add.button('Exit', pygame_menu.events.EXIT)

    def switch_to_load_model(self):
        """Switch to load model scene."""
        self.scene_manager.set_scene(LoadModelScene(self.screen, self.scene_manager))

    def switch_to_settings(self):
        """Switch to settings scene."""
        self.scene_manager.set_scene(SettingsScene(self.screen, self.scene_manager))

    def start_simulation(self):
        """Confirm before starting simulation."""
        confirmation_menu = pygame_menu.Menu('Confirm Simulation', WINDOW_SIZE[0], WINDOW_SIZE[1], theme=THEME)
        
        # Добавляем заголовок для настроек
        confirmation_menu.add.label("Simulation Settings:", align=ALIGN_LEFT)
        
        # Форматируем настройки как список
        for key, value in self.scene_manager.settings.items():
            formatted_line = f"{key}: {value}"
            if len(formatted_line) > 50:
                formatted_line = formatted_line[:47] + "..."
            confirmation_menu.add.label(formatted_line, align=ALIGN_LEFT, font_size=20)
        
        # Добавляем информацию о пути к модели
        confirmation_menu.add.label(f"Model Path: {self.scene_manager.model_path or 'Default'}", align=ALIGN_LEFT)
        
        # Кнопки для запуска или возврата
        confirmation_menu.add.button('Start', lambda: self.scene_manager.set_scene(SimulationScene(self.screen, self.scene_manager)))
        confirmation_menu.add.button('Back', lambda: self.scene_manager.set_scene(MainMenuScene(self.screen, self.scene_manager)))
        
        # Переключаемся на сцену с меню подтверждения
        self.scene_manager.set_scene(Scene(self.screen, self.scene_manager, menu=confirmation_menu))

class LoadModelScene(Scene):
    """Load model scene."""
    def __init__(self, screen, scene_manager):
        super().__init__(screen, scene_manager)
        self.model_path = scene_manager.model_path
        self.menu = pygame_menu.Menu('Load Pre-trained Model', WINDOW_SIZE[0], WINDOW_SIZE[1], theme=THEME)
        self.path_input = self.menu.add.text_input('Model Path: ', default=self.model_path, onchange=self.update_model_path)
        self.menu.add.button('Confirm', self.confirm_load_model)
        self.menu.add.button('Back', self.switch_to_main)
        self.error_label = self.menu.add.label('', label_id='error_label')

    def update_model_path(self, value):
        """Update model path."""
        self.model_path = value
        self.scene_manager.model_path = value
        self.error_label.set_title('')

    def confirm_load_model(self):
        """Confirm model loading."""
        if not self.model_path or not self.model_path.endswith('.pth'):
            self.error_label.set_title('Invalid model path', font_color=(255, 0, 0))
        else:
            self.switch_to_main()

    def switch_to_main(self):
        """Return to main menu."""
        self.scene_manager.set_scene(MainMenuScene(self.screen, self.scene_manager))

class SettingsScene(Scene):
    """Settings scene."""
    def __init__(self, screen, scene_manager):
        super().__init__(screen, scene_manager)
        self.settings = scene_manager.settings.copy()
        self.menu = pygame_menu.Menu('Adjust Settings', WINDOW_SIZE[0], WINDOW_SIZE[1], theme=THEME)
        self.error_label = self.menu.add.label('', label_id='error_label')

        # Add input fields for all settings
        self.input_fields = {}
        for key, value in self.settings.items():
            # Determine input type based on the setting
            if key in ['width', 'height', 'dish_radius', 'organism_radius', 'food_radius',
                       'obstacle_radius', 'food_quantity', 'obstacle_quantity', 'grid_size',
                       'episode_length', 'episodes']:
                input_type = INPUT_INT
                default_value = str(int(value)) if isinstance(value, (int, float)) else str(value)
            elif key == 'render':
                input_type = INPUT_TEXT
                default_value = str(value).lower()
            else:
                input_type = INPUT_FLOAT
                default_value = str(float(value)) if isinstance(value, (int, float)) else str(value)

            self.input_fields[key] = self.menu.add.text_input(
                f'{key}: ',
                default=default_value,
                input_type=input_type,
                onchange=lambda val, k=key: self.update_setting(k, val)
            )
        self.menu.add.button('Save Settings', self.save_settings)
        self.menu.add.button('Reset to Default', self.reset_to_default)
        self.menu.add.button('Back', self.switch_to_main)

    def update_setting(self, key, value):
        """Update setting value."""
        try:
            if key in ['width', 'height', 'dish_radius', 'organism_radius', 'food_radius',
                       'obstacle_radius', 'food_quantity', 'obstacle_quantity', 'grid_size',
                       'episode_length', 'episodes']:
                self.settings[key] = int(value) if value.strip() else 0
            elif key == 'render':
                self.settings[key] = value.lower() == 'true'
            else:
                self.settings[key] = float(value) if value.strip() else 0.0
            self.error_label.set_title('')
        except ValueError:
            self.error_label.set_title(f'Invalid value for {key}', font_color=(255, 0, 0))

    def save_settings(self):
        """Save settings."""
        try:
            self.scene_manager.settings = self.settings.copy()
            self.switch_to_main()
        except Exception as e:
            self.error_label.set_title(f'Error saving settings: {str(e)}', font_color=(255, 0, 0))

    def reset_to_default(self):
        """Reset settings to default."""
        self.settings = default_settings.copy()
        for key, field in self.input_fields.items():
            if key in ['width', 'height', 'dish_radius', 'organism_radius', 'food_radius',
                       'obstacle_radius', 'food_quantity', 'obstacle_quantity', 'grid_size',
                       'episode_length', 'episodes']:
                field.set_value(str(int(self.settings[key])))
            elif key == 'render':
                field.set_value(str(self.settings[key]).lower())
            else:
                field.set_value(str(float(self.settings[key])))
        self.error_label.set_title('Settings reset to default')

    def switch_to_main(self):
        """Return to main menu."""
        self.scene_manager.set_scene(MainMenuScene(self.screen, self.scene_manager))

class SimulationScene(Scene):
    """Simulation scene."""
    def __init__(self, screen, scene_manager):
        super().__init__(screen, scene_manager)
        self.settings = scene_manager.settings
        self.model_path = scene_manager.model_path
        client_settings.update(self.settings)
        self.output_dir = ".\output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.screen = screen
        self.env = RenderableEnvironment(client_settings, screen)
        self.state_dim = len(self.env.reset())
        self.action_dim = 2
        self.agent = DDPGAgent(self.state_dim, self.action_dim, client_settings)
        if self.model_path:
            self.agent.load(self.model_path)
        else:
            self.agent.load_best(self.model_path)
        self.plotter = Plotter(max_data_points=500)
        self.log_messages = []
        self.clock = pygame.time.Clock()
        self.total_rewards = []
        self.running_avg_rewards = []
        self.episode = 0
        self.state = self.env.reset()
        self.episode_reward = 0
        self.done = False
        self.step = 0
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []
        # Add back to menu button
        self.menu = pygame_menu.Menu('Simulation', WINDOW_SIZE[0], WINDOW_SIZE[1], theme=THEME)
        self.menu.add.button('Back to Menu', self.switch_to_main)
        self.menu.disable()  # Disabled by default, enable on key press

    def switch_to_main(self):
        """Return to main menu."""
        writer.close()
        self.scene_manager.set_scene(MainMenuScene(self.screen, self.scene_manager))

    def handle_event(self, event):
        """Handle events."""
        if event.type == pygame.QUIT:
            writer.close()
            pygame.quit()
            sys.exit()
        elif event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            self.env.scale = min(event.w / client_settings['width'], event.h / client_settings['height'])
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.menu.enable() if not self.menu.is_enabled() else self.menu.disable()
        if self.menu.is_enabled():
            super().handle_event(event)

    def update(self):
        """Update simulation."""
        if self.menu.is_enabled():
            return  # Pause simulation if menu is active

        if self.episode >= client_settings['episodes']:
            self.switch_to_main()
            return

        if not self.done and self.step < client_settings['episode_length']:
            action = self.agent.act(self.state)
            next_state, reward, self.done = self.env.step(action, track_approach=True)
            self.agent.remember(self.state, action, reward, next_state, self.done)
            losses = self.agent.replay()
            self.state = next_state
            self.episode_reward += reward
            self.step += 1

            if client_settings['render']:
                self.clock.tick(FPS)

            if losses:
                writer.add_scalar("Loss/Critic", losses[0], self.episode)
                writer.add_scalar("Loss/Actor", losses[1], self.episode)
                writer.add_scalar("Q_Value/Mean", losses[2], self.episode)
                # Собираем данные для усреднения
                self.critic_losses.append(losses[0])
                self.actor_losses.append(losses[1])
                self.q_values.append(losses[2])

        else:
            # End of episode
            self.total_rewards.append(self.episode_reward)
            writer.add_scalar("Reward/Episode", self.episode_reward, self.episode)
            if len(self.total_rewards) >= 50:
                running_avg = np.mean(self.total_rewards[-50:])
                self.running_avg_rewards.append(running_avg)
                writer.add_scalar("Reward/Running_Avg_50", running_avg, self.episode)

            # Вычисляем средние значения для потерь и Q-значений
            avg_critic_loss = np.mean(self.critic_losses) if self.critic_losses else 0
            avg_actor_loss = np.mean(self.actor_losses) if self.actor_losses else 0
            avg_q_value = np.mean(self.q_values) if self.q_values else 0

            # Обновляем plotter один раз за эпизод
            self.plotter.update(
                reward=self.episode_reward,
                critic_loss=avg_critic_loss,
                actor_loss=avg_actor_loss,
                q_value=avg_q_value
            )

            if self.episode % 10 == 0:
                avg_reward = np.mean(self.total_rewards[-10:]) if self.total_rewards else 0
                log_msg = f"Episode {self.episode}, Reward: {self.episode_reward:.2f}, Avg Reward: {avg_reward:.2f}"
                logging.info(log_msg)
                self.log_messages.append(log_msg)
                print(log_msg)

            if self.episode % 50 == 0 and self.episode > 0:
                test_avg_reward = test_policy(self.agent, self.env)
                writer.add_scalar("Reward/Test_Avg", test_avg_reward, self.episode)
                model_path = os.path.join(self.output_dir, f"model_ep{self.episode}_reward{test_avg_reward:.2f}.pth")
                self.agent.save(self.episode, test_avg_reward, model_path)

                if len(self.running_avg_rewards) >= 100:
                    recent_avg = np.mean(self.running_avg_rewards[-50:])
                    past_avg = np.mean(self.running_avg_rewards[-100:-50])
                    if abs(recent_avg - past_avg) / (past_avg + 1e-6) < 0.05:
                        log_msg = f"Possible plateau detected at episode {self.episode}: Recent Avg={recent_avg:.2f}, Past Avg={past_avg:.2f}"
                        logging.info(log_msg)
                        self.log_messages.append(log_msg)
                        print(log_msg)

            # Reset plotter data
            self.critic_losses = []
            self.actor_losses = []
            self.q_values = []

            self.episode += 1
            self.env = RenderableEnvironment(client_settings, self.screen, self.episode)
            self.state = self.env.reset()
            self.episode_reward = 0
            self.done = False
            self.step = 0

    def draw(self):
        """Draw simulation."""
        if self.menu.is_enabled():
            self.menu.draw(self.screen)
        else:
            if client_settings['render']:
                self.env.render(self.plotter, self.log_messages)
        pygame.display.flip()

class SceneManager:
    """Scene manager for handling transitions."""
    def __init__(self, screen):
        self.screen = screen
        self.current_scene = MainMenuScene(screen, self)
        self.settings = default_settings.copy()
        self.model_path = ""

    def set_scene(self, scene):
        """Set a new scene."""
        self.current_scene = scene

    async def run(self):
        """Main application loop."""
        clock = pygame.time.Clock()
        running = True
        while running:
            self.screen.fill((255, 255, 255))  # White background
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.current_scene = MainMenuScene(self.screen, self)
                self.current_scene.handle_event(event)
            self.current_scene.update()
            self.current_scene.draw()
            clock.tick(FPS)
            await asyncio.sleep(1.0 / FPS)
        writer.close()
        pygame.quit()

async def main():
    # Initialize screen
    base_width, base_height = client_settings['width'], client_settings['height']
    screen = pygame.display.set_mode((base_width, base_height), pygame.RESIZABLE)
    pygame.display.set_caption("Micro Evolution with DDPG")

    # Run menu
    menu = SceneManager(screen)
    await menu.run()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())