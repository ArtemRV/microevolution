import os
import re
import json
import torch
from common.utils import logging

class ModelLoader:
    """Handles loading of random, best, and custom models for DDPGAgent."""
    
    def __init__(self, settings):
        self.settings = settings
        self.output_dir = settings['output_dir']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def get_model_files(self):
        """Returns a list of model base names and their associated actor/critic paths."""
        if not self.output_dir or not os.path.exists(self.output_dir):
            return [('No models found', None)]
        
        # Collect all .pth files
        files = [f for f in os.listdir(self.output_dir) if f.endswith('_actor.pth') or f.endswith('_critic.pth')]
        
        # Group by base name (remove _actor.pth or _critic.pth)
        model_groups = {}
        for f in files:
            base_name = f.replace('_actor.pth', '').replace('_critic.pth', '')
            if base_name not in model_groups:
                model_groups[base_name] = {'actor': None, 'critic': None}
            if f.endswith('_actor.pth'):
                model_groups[base_name]['actor'] = os.path.join(self.output_dir, f)
            elif f.endswith('_critic.pth'):
                model_groups[base_name]['critic'] = os.path.join(self.output_dir, f)
        
        # Create list of tuples (base_name, {'actor': path, 'critic': path})
        result = [(name, paths) for name, paths in model_groups.items() if paths['actor'] and paths['critic']]
        
        def sort_key(item):
            name = item[0]
            match = re.match(r'model_ep(\d+)_reward([-]?\d*\.?\d*)', name)
            if match:
                epoch = int(match.group(1))
                reward = float(match.group(2))
                return (-epoch, -reward)
            return (0, 0)
        
        return sorted(result, key=sort_key) or [('No models found', None)]
    
    def load_model(self, agent, model_type, model_path=None):
        """Loads a model into the agent based on model_type ('random', 'best', 'custom')."""
        try:
            if model_type == 'random':
                logging.info("Using random model (no loading required)")
                return True
            
            elif model_type == 'best':
                best_model_path = os.path.join(self.output_dir, 'best_model_info.json')
                if not os.path.exists(best_model_path):
                    logging.warning("No best model info found")
                    return False
                with open(best_model_path, 'r') as f:
                    best_model_info = json.load(f)
                actor_path = best_model_info.get('actor_path')
                critic_path = best_model_info.get('critic_path')
                if not (actor_path and critic_path and os.path.exists(actor_path) and os.path.exists(critic_path)):
                    logging.warning("Best model paths invalid or missing")
                    return False
                self._load_model_files(agent, actor_path, critic_path)
                return True
            
            elif model_type == 'custom':
                if not model_path or not (model_path['actor'] and model_path['critic']):
                    logging.warning("Invalid custom model paths")
                    return False
                if not (os.path.exists(model_path['actor']) and os.path.exists(model_path['critic'])):
                    logging.warning(f"Custom model files not found: {model_path}")
                    return False
                self._load_model_files(agent, model_path['actor'], model_path['critic'])
                return True
            
            else:
                logging.error(f"Unknown model type: {model_type}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            return False
    
    def _load_model_files(self, agent, actor_path, critic_path):
        """Loads actor and critic models into the agent."""
        agent.actor.load_state_dict(torch.load(actor_path, weights_only=True))
        agent.critic.load_state_dict(torch.load(critic_path, weights_only=True))
        agent.actor_target.load_state_dict(agent.actor.state_dict())
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        logging.info(f"Loaded model from {actor_path} and {critic_path}")
    
    def get_info_text(self, model_type, model_path=None):
        """Returns display text for the model information."""
        if model_type == 'random':
            return "Model:\tRandom Model", \
                   ["Actor path:\tUsing random model",
                    "Critic path:\tUsing random model"]
        elif model_type == 'best':
            best_model_path = os.path.join(self.output_dir, 'best_model_info.json')
            if not os.path.exists(best_model_path):
                return "Model:\tBest Model", \
                       ["Actor path:\tNot found",
                        "Critic path:\tNot found"]
            with open(best_model_path, 'r') as f:
                best_model_info = json.load(f)
            actor_path = best_model_info.get('actor_path', 'Not found')
            critic_path = best_model_info.get('critic_path', 'Not found')
            return "Model:\tBest Model", \
                   [f"Actor path:\t{actor_path}",
                    f"Critic path:\t{critic_path}"]
        else:  # custom
            if model_path and model_path['actor'] and model_path['critic']:
                base_name = os.path.basename(model_path['actor']).replace('_actor.pth', '')
                return f"Model:\t{base_name}", \
                       [f"Actor path:\t{model_path['actor']}",
                        f"Critic path:\t{model_path['critic']}"]
            return "Model:\tNone", \
                   ["Actor path:\tNot selected",
                    "Critic path:\tNot selected"]