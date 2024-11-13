import os
import torch
import torch.nn as nn
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DQN, self).__init__()
        
        # Создаем список линейных слоев
        layers = []
        current_dim = input_dim

        # Добавляем скрытые слои согласно `hidden_dims`
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh()) # Добавляем функцию активации
            current_dim = hidden_dim

        # Добавляем выходной слой
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Tanh())
        
        # Объединяем слои в nn.Sequential для удобства
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def get_weights(self):
        weights = [layer.weight.data.numpy() for layer in self.model if isinstance(layer, nn.Linear)]
        return weights
    
    def load_model(self, settings):
        if 'model_path' in settings and settings['model_path'] == '':
            self.load_state_dict(torch.load(settings['model_path'], weights_only=True))

    def save_model(self, settings, episode, done):
        if done and episode % settings['Episode length'] == 0:
            print(f"Episode: {episode}!")
            # Сохранение модели
            save_dir = "weights&models/models/autosave"
            filename = f"model{episode // settings['Episode length']}.pth"
            file_path = os.path.join(save_dir, filename)
            torch.save(self.state_dict(), file_path)

    def choose_action(self, state, settings, OUTPUT_DIM):
        if np.random.rand() < settings['EPSILON']:
            # Случайное действие (эксплорейшн)
            return torch.FloatTensor(1, OUTPUT_DIM).uniform_(-1, 1)
        else:
            return self.model_action(state)
            
    def model_action(self, state):
        with torch.no_grad():
            action = self(state)
            return action.view(1, -1)
        
    def replay(self, memory, optimizer, loss_fn, batch_size, settings):
        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_tensor = state.clone().detach().unsqueeze(0)  # размер (1, state_size)
            next_state_tensor = next_state.clone().detach().unsqueeze(0) # размер (1, state_size)
            target = reward
            if not done:
                # Обновление Q-значения
                target += settings['GAMMA'] * torch.max(self(next_state_tensor)).item()

            target_f = self(state_tensor).squeeze().clone()
            action_0 = action[0][0]
            action_1 = action[0][1]

            # Обновляем значения target_f с учетом action
            target_f[0] = action_0 * target  # Обновляем значение для первого выхода
            target_f[1] = action_1 * target  # Обновляем значение для второго выхода

            # Обучение модели
            optimizer.zero_grad()
            loss = loss_fn(target_f, self(state_tensor).squeeze())
            loss.backward()
            optimizer.step()

        return loss.item()