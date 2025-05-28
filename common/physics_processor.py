import numpy as np

class PhysicsProcessor:
    def __init__(self, env, settings):
        self.env = env
        self.settings = settings

    def process_generic_movement(self, game_object):
        # Update position
        game_object.pos += game_object.vel

        # Calculate distance to center
        dist_to_center = np.linalg.norm(game_object.pos - self.env.dish_center)

        # Check for collision
        if dist_to_center > self.env.dish_radius - game_object.radius:
            # Calculate normal vector (pointing from object towards dish center)
            # Note: The original GameObject.move had normal = (self.env.dish_center - self.pos) / (dist_to_center + 1e-6)
            # which is already pointing inwards (from object to center if pos is outside, or from center to object if pos is inside).
            # For reflection, we want the normal of the surface, which points from the center outwards to the object.
            # However, the reflection formula v' = v - 2 * dot(v, n) * n assumes n is the surface normal.
            # If we use a normal pointing inwards (from object to center), the reflection still works out correctly
            # because the repositioning step will place the object on the boundary.
            # Let's stick to the logic similar to GameObject.move for the normal calculation direction for now.
            normal = (self.env.dish_center - game_object.pos) / (dist_to_center + 1e-6) # Normal pointing inwards

            # Reflect velocity
            dot_product = np.dot(game_object.vel, normal)
            game_object.vel = game_object.vel - 2 * dot_product * normal

            # Reposition to be exactly on the boundary
            # Direction from center to object
            direction_from_center = (game_object.pos - self.env.dish_center) / (dist_to_center + 1e-6)
            game_object.pos = self.env.dish_center + direction_from_center * (self.env.dish_radius - game_object.radius)

    def process_organism_action(self, organism, action):
        # Calculate acceleration
        acceleration = np.array(action) * self.settings['organism']['max_acceleration']

        # Update organism's velocity
        organism.vel += acceleration

        # Calculate current speed
        speed = np.linalg.norm(organism.vel)

        # Cap speed if it exceeds max_speed
        if speed > organism.max_speed and speed > 0:
            organism.vel = organism.vel / speed * organism.max_speed

    def resolve_food_food_collisions(self, foods):
        for i, food1 in enumerate(foods):
            for food2 in foods[i+1:]: # Starts from i+1 to get unique pairs
                dist = np.linalg.norm(food1.pos - food2.pos)
                if dist < food1.radius + food2.radius and dist > 0: # Ensure dist > 0
                    direction = (food1.pos - food2.pos) / (dist + 1e-6) # Added epsilon for safety
                    overlap = food1.radius + food2.radius - dist
                    food1.pos += direction * (overlap / 2)
                    food2.pos -= direction * (overlap / 2)

    def resolve_food_obstacle_collisions(self, foods, obstacles):
        for food in foods:
            for obstacle in obstacles:
                dist = np.linalg.norm(food.pos - obstacle.pos)
                if dist < food.radius + obstacle.radius:
                    direction = (food.pos - obstacle.pos) / (dist + 1e-6) # Epsilon for safety
                    overlap = food.radius + obstacle.radius - dist
                    food.pos += direction * overlap
