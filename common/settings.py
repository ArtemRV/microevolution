# Default settings shared by client and server
default_settings = {
    'organism': {
        'radius': 10,
        'max_speed': 5,
        'initial_energy': 100,
        'max_acceleration': 1,
        'energy_per_food': 20,
        'energy_per_step': 0.05,
        'enabled': True,
        'random_start': True,
        'start_radius': 230,
    },
    'food': {
        'radius': 5,
        'spawn_rate': 0.1,
        'quantity': 20,
        'min_quantity': 10,
        'increment_quantity': 0,
        'enabled': True
    },
    'obstacle': {
        'radius': 15,
        'max_speed': 2,
        'quantity': 5,
        'min_quantity': 3,
        'increment_quantity': 0,
        'enabled': True
    },
    'general': {
        'width': 1200,
        'height': 800,
        'dish_radius': 250,
        'grid_size': 50,
    },
    "rewards": {
        "eat": {
            "value": 100.0,
            "enabled": True
        },
        "dish_collision": {
            "value": -100.0,
            "enabled": True,
            "end_episode": True
        },
        "obstacle_collision": {
            "value": -20.0,
            "enabled": False,
            "end_episode": False
        },
        "energy": {
            "value": -1,
            'increment_steps': 20,
            'increment': 0.5,
            "enabled": True
        },
        "approach": {
            "value": 5,
            "enabled": True
        },
        "survival": {
            "value": 0.5,
            'increment_steps': 50,
            'increment': 0.5,
            "enabled": True
        }
    },
}

training_settings = {
    'episodes': 3000,
    'episode_length': 500,
    'actor_lr': 5e-4,
    'critic_lr': 1e-3,
    'gamma': 0.99,
    'tau': 0.001, #0.005, # Target network update speed
    'batch_size': 128,
    'memory_size': 100000,
    'rewards_enabled': True,
    # Additional settings
    'increment_obstacle': True,
    'increment_food': True,
    'increment_episodes': 100,
}

trainer_settings = {
    **default_settings,
    **training_settings,
    'rendering_enabled': False,
    'output_dir': 'output',
    'load_model': {
        'random_model': True,
        'model_path': None,
    }
}

# Client-specific settings (extend default_settings)
client_settings = {
    **default_settings,
    **training_settings,
    'rendering_enabled': True,
    'output_dir': 'output',
    'load_model': {
        'random_model': False,
        'model_path': None,
    }
}

server_settings = {
    **default_settings,
    'rewards_enabled': False,
    'rendering_enabled': True,
    'output_dir': 'output',
    'general': {
        'width': 600,
        'height': 600,
        'dish_radius': 250,
        'grid_size': 50,
    },
}

# Colors
class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (200, 200, 200)