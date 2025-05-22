import matplotlib.pyplot as plt
import numpy as np
import os
from common.utils import logging
import pygame

class Plotter:
    def __init__(self, max_data_points=500):
        self.max_data_points = max_data_points
        self.rewards = []
        self.critic_losses = []
        self.actor_losses = []
        self.q_values = []
        self.colors = {
            'reward': (0, 128, 0),      # Green
            'critic_loss': (255, 0, 0), # Red
            'actor_loss': (0, 0, 255),  # Blue
            'q_value': (128, 0, 128)    # Purple
        }

    def update(self, reward=None, critic_loss=None, actor_loss=None, q_value=None):
        if reward is not None:
            self.rewards.append(reward)
            if len(self.rewards) > self.max_data_points:
                self.rewards.pop(0)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
            if len(self.critic_losses) > self.max_data_points:
                self.critic_losses.pop(0)
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
            if len(self.actor_losses) > self.max_data_points:
                self.actor_losses.pop(0)
        if q_value is not None:
            self.q_values.append(q_value)
            if len(self.q_values) > self.max_data_points:
                self.q_values.pop(0)

    def draw_plots(self, screen):
        width, height = screen.get_size()
        plot_width = width // 4
        plot_height = height // 4  # Сохраняем исходную высоту
        x_offset = width - plot_width - 10
        plots = [
            ('Reward', self.rewards, self.colors['reward'], 0),
            ('Losses', None, None, 1),  # Специальный случай для потерь
            ('Q-Value', self.q_values, self.colors['q_value'], 2)
        ]

        for title, data, color, idx in plots:
            # if not data:
            #     continue
            y_offset = 10 + idx * (plot_height + 10)
            pygame.draw.rect(screen, (200, 200, 200), (x_offset, y_offset, plot_width, plot_height))
            pygame.draw.rect(screen, (0, 0, 0), (x_offset, y_offset, plot_width, plot_height), 1)

            if title == 'Losses':
                # Объединяем данные для потерь
                all_losses = self.critic_losses + self.actor_losses
                if all_losses:
                    max_val = max(all_losses) if max(all_losses) > 0 else 1
                    min_val = min(all_losses) if min(all_losses) < 0 else 0
                    val_range = max_val - min_val if max_val != min_val else 1

                    # Линия для Critic Loss
                    if self.critic_losses:
                        points = []
                        for i, val in enumerate(self.critic_losses):
                            x = x_offset + (i / (len(self.critic_losses) - 1)) * (plot_width - 1) if len(self.critic_losses) > 1 else x_offset
                            y = y_offset + plot_height - ((val - min_val) / val_range) * (plot_height - 1)
                            points.append((x, y))
                        if len(points) > 1:
                            pygame.draw.lines(screen, self.colors['critic_loss'], False, points, 2)

                    # Линия для Actor Loss
                    if self.actor_losses:
                        points = []
                        for i, val in enumerate(self.actor_losses):
                            x = x_offset + (i / (len(self.actor_losses) - 1)) * (plot_width - 1) if len(self.actor_losses) > 1 else x_offset
                            y = y_offset + plot_height - ((val - min_val) / val_range) * (plot_height - 1)
                            points.append((x, y))
                        if len(points) > 1:
                            pygame.draw.lines(screen, self.colors['actor_loss'], False, points, 2)

                    # Легенда
                    font = pygame.font.SysFont(None, 18)
                    label_critic = font.render('Critic', True, self.colors['critic_loss'])
                    label_actor = font.render('Actor', True, self.colors['actor_loss'])
                    screen.blit(label_critic, (x_offset + 5, y_offset + 25))
                    screen.blit(label_actor, (x_offset + 5, y_offset + 45))
            else:
                # Обычные графики (Reward, Q-Value)
                if data:
                    max_val = max(data) if max(data) > 0 else 1
                    min_val = min(data) if min(data) < 0 else 0
                    val_range = max_val - min_val if max_val != min_val else 1

                    points = []
                    for i, val in enumerate(data):
                        x = x_offset + (i / (len(data) - 1)) * (plot_width - 1) if len(data) > 1 else x_offset
                        y = y_offset + plot_height - ((val - min_val) / val_range) * (plot_height - 1)
                        points.append((x, y))

                    if len(points) > 1:
                        pygame.draw.lines(screen, color, False, points, 2)

                font = pygame.font.SysFont(None, 20)
                label = font.render(title, True, (0, 0, 0))
                screen.blit(label, (x_offset + 5, y_offset + 5))

def plot_training_progress(total_rewards, settings):
    try:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(total_rewards, label='Episode Reward')
        if len(total_rewards) >= 50:
            running_avg = np.convolve(total_rewards, np.ones(50)/50, mode='valid')
            plt.plot(range(len(running_avg)), running_avg, label='50-Episode Avg', linestyle='--')
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/training_progress.png')
        plt.close()
        logging.info("Saved training progress plot")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")

def test_policy(agent, env, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        while not done and step < env.settings['episode_length']:
            action = agent.act(state, add_noise=False)
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            step += 1
        total_rewards.append(episode_reward)
    avg_reward = np.mean(total_rewards)
    logging.info(f"Test Policy: Avg Reward over {episodes} episodes = {avg_reward:.2f}")
    return avg_reward