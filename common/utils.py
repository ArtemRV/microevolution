import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_position_free(pos, radius, dish_center, dish_radius, objects, min_distance):
    dist_to_center = np.linalg.norm(pos - dish_center)
    if dist_to_center > dish_radius - radius:
        return False
    for obj in objects:
        dist = np.linalg.norm(pos - obj.pos)
        if dist < radius + obj.radius + min_distance:
            return False
    return True