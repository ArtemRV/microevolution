import argparse
import numpy as np
import uuid
import os
from client.ddpg import DDPGAgent
from client.plotting import test_policy
from common.core import Environment
from common.settings import client_settings
from common.utils import logging
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """Handles model training without UI."""
    def __init__(self, settings, output_dir, model_path=None, episodes=1000):
        self.settings = settings.copy()
        self.output_dir = output_dir
        self.model_path = model_path
        self.episodes = episodes
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs", f"train_{uuid.uuid4()}"))
        self.env = Environment(self.settings)
        self.state_dim = len(self.env.reset())
        self.action_dim = 2
        self.agent = DDPGAgent(self.state_dim, self.action_dim, self.settings)
        if self.model_path:
            self.agent.load(self.model_path)
        else:
            self.agent.load_best()
        self.total_rewards = []
        self.running_avg_rewards = []
        self.log_messages = []
        # self.critic_losses = []
        # self.actor_losses = []
        # self.q_values = []

    def train(self):
        """Run the training loop."""
        for episode in range(self.episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done and step < self.settings['episode_length']:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action, track_approach=True)
                self.agent.remember(state, action, reward, next_state, done)
                losses = self.agent.replay()
                state = next_state
                episode_reward += reward
                step += 1

                if losses:
                    self.writer.add_scalar("Loss/Critic", losses[0], episode)
                    self.writer.add_scalar("Loss/Actor", losses[1], episode)
                    self.writer.add_scalar("Q_Value/Mean", losses[2], episode)
                    # self.critic_losses.append(losses[0])
                    # self.actor_losses.append(losses[1])
                    # self.q_values.append(losses[2])

            self.total_rewards.append(episode_reward)
            self.writer.add_scalar("Reward/Episode", episode_reward, episode)

            if len(self.total_rewards) >= 50:
                running_avg = np.mean(self.total_rewards[-50:])
                self.running_avg_rewards.append(running_avg)
                self.writer.add_scalar("Reward/Running_Avg_50", running_avg, episode)

            # avg_critic_loss = np.mean(self.critic_losses) if self.critic_losses else 0
            # avg_actor_loss = np.mean(self.actor_losses) if self.actor_losses else 0
            # avg_q_value = np.mean(self.q_values) if self.q_values else 0

            if episode % 10 == 0:
                avg_reward = np.mean(self.total_rewards[-10:]) if self.total_rewards else 0
                log_msg = f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}"
                logging.info(log_msg)
                self.log_messages.append(log_msg)
                print(log_msg)

            if episode % 50 == 0 and episode > 0:
                test_avg_reward = test_policy(self.agent, self.env)
                self.writer.add_scalar("Reward/Test_Avg", test_avg_reward, episode)
                model_path = os.path.join(self.output_dir, f"model_ep{episode}_reward{test_avg_reward:.2f}.pth")
                self.agent.save(episode, test_avg_reward, model_path)

                if len(self.running_avg_rewards) >= 100:
                    recent_avg = np.mean(self.running_avg_rewards[-50:])
                    past_avg = np.mean(self.running_avg_rewards[-100:-50])
                    if abs(recent_avg - past_avg) / (past_avg + 1e-6) < 0.05:
                        log_msg = f"Possible plateau detected at episode {episode}: Recent Avg={recent_avg:.2f}, Past Avg={past_avg:.2f}"
                        logging.info(log_msg)
                        self.log_messages.append(log_msg)
                        print(log_msg)

            # self.critic_losses = []
            # self.actor_losses = []
            # self.q_values = []

        self.writer.close()
        final_model_path = os.path.join(self.output_dir, "final_model.pth")
        self.agent.save(self.episodes, np.mean(self.total_rewards[-10:]), final_model_path)
        print(f"Training completed. Final model saved at: {final_model_path}")

def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Micro Evolution Training with DDPG")
    parser.add_argument('--output-dir', type=str, default='./output', help="Directory to save models and logs")
    parser.add_argument('--model-path', type=str, help="Path to pre-trained model")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--episode-length', type=int, help="Length of each episode")
    parser.add_argument('--width', type=int, help="Environment width")
    parser.add_argument('--height', type=int, help="Environment height")
    parser.add_argument('--dish-radius', type=int, help="Dish radius")
    parser.add_argument('--food-quantity', type=int, help="Number of food items")
    parser.add_argument('--obstacle-quantity', type=int, help="Number of obstacles")
    return parser.parse_args()

def main():
    """CLI entry point for training."""
    args = parse_args()
    settings = client_settings.copy()
    if args.episode_length:
        settings['episode_length'] = args.episode_length
    if args.width:
        settings['width'] = args.width
    if args.height:
        settings['height'] = args.height
    if args.dish_radius:
        settings['dish_radius'] = args.dish_radius
    if args.food_quantity:
        settings['food_quantity'] = args.food_quantity
    if args.obstacle_quantity:
        settings['obstacle_quantity'] = args.obstacle_quantity
    settings['render'] = False  # Disable rendering
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = Trainer(settings, args.output_dir, args.model_path, args.episodes)
    trainer.train()

if __name__ == "__main__":
    main()