# Default settings shared by client and server
default_settings = {
    'organism': {
        'radius': 10,
        'max_speed': 5,
        'initial_energy': 100,
        'max_acceleration': 1,
        'energy_per_food': 20,
        'energy_per_step': 0.05,
        'enabled': True
    },
    'food': {
        'radius': 5,
        'spawn_rate': 0.1,
        'quantity': 20,
        'enabled': True
    },
    'obstacle': {
        'radius': 15,
        'max_speed': 2,
        'quantity': 5,
        'enabled': True
    },
    'general': {
        'width': 1200,
        'height': 800,
        'dish_radius': 250,
        'grid_size': 50,
    }
}

training_settings = {
    'episodes': 1000,
    'episode_length': 500,
    'actor_lr': 5e-4,
    'critic_lr': 1e-3,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 128,
    'memory_size': 100000,
    'rewards_enabled': True,
}

trainer_settings = {
    **default_settings,
    **training_settings,
    'random_model': True,
    'rendering_enabled': False,
    'output_dir': 'output',
    'model_path': None,
}

# Client-specific settings (extend default_settings)
client_settings = {
    **default_settings,
    **training_settings,
    'random_model': True,
    'rendering_enabled': True,
    'output_dir': 'output',
}

server_settings = {
    **default_settings,
    'rewards_enabled': False,
    'rendering_enabled': True,
    'output_dir': 'output',
}

# Colors
class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (200, 200, 200)