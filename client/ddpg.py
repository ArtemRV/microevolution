import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import random
import os
import json
import numpy as np
from common.models import Actor, Critic
from common.utils import logging
from client.model_loader import ModelLoader

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.1, decay=0.995):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.decay = decay
        self.initial_sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.sigma = self.initial_sigma

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        self.sigma *= self.decay
        return self.state

class DDPGAgent:
    def __init__(self, state_dim, action_dim, settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=settings['actor_lr'], weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=settings['critic_lr'], weight_decay=1e-4)
        self.memory = deque(maxlen=settings['memory_size'])
        self.batch_size = settings['batch_size']
        self.gamma = settings['gamma']
        self.tau = settings['tau']
        self.noise = OUNoise(action_dim, sigma=0.1, decay=0.995)
        self.settings = settings
        self.best_test_reward = float('-inf')
        self.best_model_info = {}
        self.model_loader = ModelLoader(settings)

    def act(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        try:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert lists/tuples to NumPy arrays first
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.FloatTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

            next_actions = self.actor_target(next_states)
            noise = torch.clamp(torch.randn_like(next_actions) * 0.1, -0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, -1, 1)

            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            current_q = self.critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            predicted_actions = self.actor(states)
            actor_loss = -self.critic(states, predicted_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

            return critic_loss.item(), actor_loss.item(), current_q.mean().item()
        except Exception as e:
            logging.error(f"Error in replay: {e}")
            return None

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, episode, test_reward, model_path):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            actor_path = model_path.replace(".pth", "_actor.pth")
            critic_path = model_path.replace(".pth", "_critic.pth")
            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)
            logging.info(f"Saved models for episode {episode} at {actor_path}, {critic_path}")

            if test_reward > self.best_test_reward:
                self.best_test_reward = test_reward
                best_actor_path = os.path.join(os.path.dirname(model_path), "best_actor.pth")
                best_critic_path = os.path.join(os.path.dirname(model_path), "best_critic.pth")
                torch.save(self.actor.state_dict(), best_actor_path)
                torch.save(self.critic.state_dict(), best_critic_path)
                self.best_model_info = {
                    'episode': episode,
                    'test_reward': test_reward,
                    'actor_path': best_actor_path,
                    'critic_path': best_critic_path,
                    'settings': self.settings
                }
                with open(os.path.join(os.path.dirname(model_path), "best_model_info.json"), 'w') as f:
                    json.dump(self.best_model_info, f, indent=4)
                logging.info(f"New best model saved with test reward {test_reward:.2f} at episode {episode}")
        except Exception as e:
            logging.error(f"Failed to save models: {e}")

    def load_model(self, model_path):
        """Loads a custom model using ModelLoader."""
        return self.model_loader.load_model(self, 'custom', model_path)

    def load_best_model(self):
        """Loads the best model using ModelLoader."""
        return self.model_loader.load_model(self, 'best')