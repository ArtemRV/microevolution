import argparse
import numpy as np
import uuid
import os
from client.ddpg import DDPGAgent
from client.plotting import test_policy
from common.game_object import Environment
from common.settings import trainer_settings
from common.utils import logging
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    """Handles model training without UI."""
    def __init__(self, settings):
        self.settings = settings.copy()
        self.output_dir = self.settings['output_dir']
        self.model_path = self.settings['model_path']
        self.episodes = self.settings['episodes']
        self.random_model = self.settings['random_model']
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "runs", f"train_{uuid.uuid4()}"))
        self.env = Environment(self.settings)
        self.state_dim = len(self.env.reset())
        self.action_dim = 2
        self.agent = DDPGAgent(self.state_dim, self.action_dim, self.settings)
        if not self.random_model:
            if self.model_path:
                self.agent.load(self.model_path)
            else:
                self.agent.load_best(self.model_path)
        self.total_rewards = []
        self.running_avg_rewards = []
        self.log_messages = []

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

            self.total_rewards.append(episode_reward)
            self.writer.add_scalar("Reward/Episode", episode_reward, episode)

            if len(self.total_rewards) >= 50:
                running_avg = np.mean(self.total_rewards[-50:])
                self.running_avg_rewards.append(running_avg)
                self.writer.add_scalar("Reward/Running_Avg_50", running_avg, episode)

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

        self.writer.close()
        final_model_path = os.path.join(self.output_dir, "final_model.pth")
        self.agent.save(self.episodes, np.mean(self.total_rewards[-10:]), final_model_path)
        print(f"Training completed. Final model saved at: {final_model_path}")

def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Micro Evolution Training with DDPG")
    parser.add_argument('--output-dir', type=str, help="Directory to save models and logs")
    parser.add_argument('--model-path', type=str, help="Path to pre-trained model")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--episode-length', type=int, help="Length of each episode")
    parser.add_argument('--dish-radius', type=int, help="Dish radius")
    parser.add_argument('--food-quantity', type=int, help="Number of food items")
    parser.add_argument('--obstacle-quantity', type=int, help="Number of obstacles")
    parser.add_argument('--random-model', action='store_false', help="Start with a randomly initialized model, default=True")
    return parser.parse_args()

def main():
    """CLI entry point for training."""
    args = parse_args()
    print(args)
    settings = trainer_settings.copy()
    if args.output_dir:
        settings['output_dir'] = args.output_dir
    if args.model_path:
        settings['model_path'] = args.model_path
    if args.episodes:
        settings['episodes'] = args.episodes
    if args.episode_length:
        settings['episode_length'] = args.episode_length
    if args.dish_radius:
        settings['general']['dish_radius'] = args.dish_radius
    if args.food_quantity:
        settings['food']['quantity'] = args.food_quantity
    if args.obstacle_quantity:
        settings['obstacle']['quantity'] = args.obstacle_quantity
    if args.random_model:
        settings['random_model'] = args.random_model
    os.makedirs(settings['output_dir'], exist_ok=True)
    trainer = Trainer(settings)
    trainer.train()

if __name__ == "__main__":
    main()